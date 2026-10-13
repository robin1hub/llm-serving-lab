"""单卡物理KV块、请求块表与事务生命周期；参考师兄职责划分，见第12章。"""

from dataclasses import dataclass, replace
from typing import Literal

from .scheduler import BatchPlan, ReadyTask


@dataclass(frozen=True)
class KVConfig:
    gpu_id: str
    total_blocks: int
    block_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_id, str) or not self.gpu_id.strip():
            raise ValueError("kv_cache.gpu_id 必须是非空字符串")
        for name in ("total_blocks", "block_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"kv_cache.{name} 必须是正整数")


@dataclass(frozen=True)
class KVBlock:
    block_id: int
    gpu_id: str
    logical_index: int
    valid_start: int
    valid_end: int
    state: Literal["Reserved", "Populating", "Ready"]
    request_refs: tuple[str, ...]
    execution_owner: int | None = None


@dataclass(frozen=True)
class RequestBlockTable:
    request_id: str
    gpu_id: str
    block_ids: tuple[int, ...]
    cached_tokens: int


@dataclass(frozen=True)
class KVSnapshot:
    gpu_id: str
    total_blocks: int
    free_blocks: int
    peak_used_blocks: int
    internal_fragmentation_tokens: int
    blocks: tuple[KVBlock, ...]
    tables: tuple[RequestBlockTable, ...]
    reservations: tuple["KVReservation", ...]


@dataclass(frozen=True)
class KVRecord:
    time_ms: float
    action: str
    snapshot: KVSnapshot


@dataclass(frozen=True)
class KVWrite:
    request_id: str
    old_tokens: int
    new_tokens: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class KVReservation:
    gpu_id: str
    transaction_id: int
    writes: tuple[KVWrite, ...]
    new_block_ids: tuple[int, ...]
    previous_blocks: tuple[KVBlock, ...]


class BlockPool:
    """每次模拟创建一个池。保留块身份与GPU归属，不模拟真实KV数组或地址。"""

    def __init__(self, config: KVConfig) -> None:
        self.config = config
        self._blocks: dict[int, KVBlock] = {}
        self._tables: dict[str, RequestBlockTable] = {}
        self._transactions: dict[int, tuple[KVReservation, str]] = {}
        self._next_block = 1
        self._next_transaction = 1
        self._peak = 0

    @property
    def free_blocks(self) -> int:
        return self.config.total_blocks - len(self._blocks)

    def cached_tokens(self, request_id: str) -> int:
        table = self._tables.get(request_id)
        return table.cached_tokens if table else 0

    def additional_blocks(self, request_id: str, tokens: int) -> int:
        table = self._tables.get(request_id)
        allocated = len(table.block_ids) if table else 0
        return self._block_count(self.cached_tokens(request_id) + tokens) - allocated

    def _block_count(self, tokens: int) -> int:
        return (tokens + self.config.block_tokens - 1) // self.config.block_tokens

    def admit(self, candidate: BatchPlan) -> BatchPlan | None:
        # 【重点1：只读容量准入】先保留可容纳Decode，再用尾块和剩余块缩小Prefill。
        # 不改变原计划，也不分配；算法上限已由Scheduler限制，准入只会删减。
        free = self.free_blocks
        selected: list[ReadyTask] = []
        for item in candidate.items:
            if item.phase != "Decode":
                continue
            required = self.additional_blocks(item.request_id, item.token_count)
            if required <= free:
                selected.append(item)
                free -= required
        for item in candidate.items:
            if item.phase != "Prefill":
                continue
            table = self._tables.get(item.request_id)
            allocated = len(table.block_ids) if table else 0
            room = (allocated + free) * self.config.block_tokens - self.cached_tokens(item.request_id)
            tokens = min(item.token_count, room)
            if tokens > 0:
                selected.append(replace(item, token_count=tokens))
                free -= self.additional_blocks(item.request_id, tokens)
        if not selected:
            return None
        # 纯Decode维持顺序；混合仍Decode在前，Prefill在后。
        items = tuple(selected)
        reason = candidate.reason if items == candidate.items else candidate.reason + "；KV容量删减/缩块"
        return BatchPlan(items, reason)

    def reserve(self, plan: BatchPlan) -> KVReservation:
        # 【重点2：先完整校验再修改】容量失败不会留下半个batch的分配。
        if not plan.items or len(set(plan.request_ids)) != len(plan.items):
            raise ValueError("KV预留要求非空且请求不重复")
        if self._transactions:
            raise ValueError("当前单GPU池只允许一个在途事务")
        required = 0
        for item in plan.items:
            if type(item.token_count) is not int or item.token_count <= 0:
                raise ValueError("KV写入量必须为正整数")
            if item.phase == "Prefill" and item.range_start != self.cached_tokens(item.request_id):
                raise ValueError("Prefill范围必须紧接已缓存输入")
            required += self.additional_blocks(item.request_id, item.token_count)
        if required > self.free_blocks:
            raise ValueError("KV容量不足，预留未发生")
        transaction_id = self._next_transaction
        self._next_transaction += 1
        writes: list[KVWrite] = []
        new_ids: list[int] = []
        previous: list[KVBlock] = []
        for item in plan.items:
            old = self.cached_tokens(item.request_id)
            table = self._tables.get(item.request_id)
            ids = list(table.block_ids if table else ())
            new_end = old + item.token_count
            for logical in range(self._block_count(old), self._block_count(new_end)):
                block_id = self._next_block
                self._next_block += 1
                start = logical * self.config.block_tokens
                self._blocks[block_id] = KVBlock(block_id, self.config.gpu_id, logical,
                    start, start, "Reserved", (item.request_id,), transaction_id)
                ids.append(block_id)
                new_ids.append(block_id)
            # 只锁住实际写入的尾块/新块；现有表与cached_tokens在完成前不改变。
            for block_id in ids:
                block = self._blocks[block_id]
                if block.valid_start < new_end and block.valid_start + self.config.block_tokens > old:
                    if block_id not in new_ids:
                        previous.append(block)
                        self._blocks[block_id] = replace(block, execution_owner=transaction_id)
            writes.append(KVWrite(item.request_id, old, new_end, tuple(ids)))
        reservation = KVReservation(self.config.gpu_id, transaction_id, tuple(writes),
                                    tuple(new_ids), tuple(previous))
        self._transactions[transaction_id] = (reservation, "Reserved")
        self._peak = max(self._peak, len(self._blocks))
        return reservation

    def _check(self, reservation: KVReservation, state: str | None = None) -> None:
        saved = self._transactions.get(reservation.transaction_id)
        if saved is None or saved[0] != reservation or (state is not None and saved[1] != state):
            raise ValueError("过期KV事务或生命周期状态不匹配")

    def begin_populating(self, reservation: KVReservation) -> None:
        self._check(reservation, "Reserved")
        for block_id, block in tuple(self._blocks.items()):
            if block.execution_owner == reservation.transaction_id:
                self._blocks[block_id] = replace(block, state="Populating")
        self._transactions[reservation.transaction_id] = (reservation, "Populating")

    def complete(self, reservation: KVReservation) -> None:
        # 【重点3：完成后提交范围】填充期间valid_end仍是旧范围，不能提前声称数据就绪。
        self._check(reservation, "Populating")
        for write in reservation.writes:
            for block_id in write.block_ids:
                block = self._blocks[block_id]
                if block.execution_owner == reservation.transaction_id:
                    self._blocks[block_id] = replace(block, state="Ready", execution_owner=None,
                        valid_end=min(write.new_tokens, block.valid_start + self.config.block_tokens))
            self._tables[write.request_id] = RequestBlockTable(write.request_id, self.config.gpu_id,
                                                               write.block_ids, write.new_tokens)
        del self._transactions[reservation.transaction_id]

    def rollback(self, reservation: KVReservation) -> None:
        self._check(reservation)
        for block_id in reservation.new_block_ids:
            del self._blocks[block_id]
        for block in reservation.previous_blocks:
            self._blocks[block.block_id] = block
        del self._transactions[reservation.transaction_id]

    def release_request(self, request_id: str) -> None:
        # 【重点4：保护生命周期】正在写入时禁止释放。其他请求引用未来也会阻止回收。
        referenced = [block for block in self._blocks.values() if request_id in block.request_refs]
        if any(block.execution_owner is not None for block in referenced):
            raise ValueError("请求KV仍在执行，不能释放")
        for block in referenced:
            refs = tuple(ref for ref in block.request_refs if ref != request_id)
            if refs:
                self._blocks[block.block_id] = replace(block, request_refs=refs)
            else:
                del self._blocks[block.block_id]
        self._tables.pop(request_id, None)

    def snapshot(self) -> KVSnapshot:
        # 所有元素冻结；历史不会被后续填充/释放修改。当前只统计已就绪块尾部浪费。
        fragmentation = sum(self.config.block_tokens - (b.valid_end - b.valid_start)
                            for b in self._blocks.values() if b.state == "Ready")
        return KVSnapshot(self.config.gpu_id, self.config.total_blocks, self.free_blocks,
                          self._peak, fragmentation, tuple(self._blocks.values()),
                          tuple(self._tables.values()), tuple(value[0] for value in self._transactions.values()))

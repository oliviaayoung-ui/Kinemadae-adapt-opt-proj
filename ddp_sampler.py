import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)
class CustomDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.current_index = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        while self.current_index < len(indices):
            yield indices[self.current_index]
            self.current_index += 1
        self.current_index = 0
        
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
    
    def state_dict(self) -> dict:
        return {
            'epoch': self.epoch,
            'seed': self.seed,
            'current_index': self.current_index
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.epoch = state_dict['epoch']
        self.seed = state_dict['seed']
        self.current_index = state_dict.get('current_index', 0)


class MixedLengthBatchSampler(Sampler):
    """CustomDistributedSampler 위에서 배치마다 길이(17/81)+배치크기를 seed로 결정하는 batch_sampler.
       - 모든 rank 가 같은 batch_idx -> 같은 (use81, L, bs) (seed 에 rank 안 넣음) -> DDP shape 동기화.
       - index 는 rank별 다름 (base_sampler 가 rank shard) -> DDP 답게 데이터는 다르고 shape 만 같음.
       - 전 rank 가 같은 indices 수 + 같은 bs 소비 -> 같은 배치 수 -> hang 없음.
       Open-Sora VariableVideoBatchSampler(sampler.py:194) 의 간소판 (해상도 고정, 17/81 2-bucket).
       yield 하는 "{idx}-{L}" 는 video_dataset.TrainVideoDataset.__getitem__ 이 파싱."""

    def __init__(self, base_sampler, batch_size, base_num_frames,
                 mix_81_prob, mix_81_num_frames, mix_81_batch_size, seed=0, drop_last=True):
        self.base = base_sampler            # CustomDistributedSampler (rank별 shard)
        self.batch_size = batch_size        # 17f 배치 크기
        self.f17 = base_num_frames          # 기본 프레임수 (17)
        self.prob = mix_81_prob             # 81f 가 뽑힐 확률
        self.f81 = mix_81_num_frames        # 81
        self.bs81 = mix_81_batch_size       # 81f 전용 배치 크기 (메모리 때문에 작게)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)      # base 도 같은 epoch (셔플 동기화)

    def __iter__(self) -> Iterator:
        indices = list(self.base)           # 이 rank 의 epoch 인덱스 (rank별 다름, 길이는 전 rank 동일)
        i, b = 0, 0
        while i < len(indices):
            # rank 안 넣음 -> 모든 rank 가 같은 (use81, L, bs) -> 같은 step 에 같은 shape
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1_000_003 + b)
            use81 = torch.rand(1, generator=g).item() < self.prob
            L = self.f81 if use81 else self.f17
            bs = self.bs81 if use81 else self.batch_size
            chunk = indices[i:i + bs]
            i += bs
            b += 1
            if len(chunk) < bs and self.drop_last:
                break                        # 마지막 partial drop (전 rank 동일 시점이라 안전)
            yield [f"{idx}-{L}" for idx in chunk]   # video_dataset 이 "{idx}-{L}" 파싱

    def __len__(self) -> int:
        avg_bs = (1.0 - self.prob) * self.batch_size + self.prob * self.bs81
        return max(1, int(len(self.base) / max(1.0, avg_bs)))   # random L 이라 근사값
        
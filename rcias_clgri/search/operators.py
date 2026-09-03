"""Shared source-faithful variation operators for literature baselines."""

from __future__ import annotations

import random
from typing import TypeVar

from rcias_clgri.data.instance import Instance


Gene = TypeVar("Gene")


def pox_pair(
    instance: Instance,
    left: tuple[str, ...],
    right: tuple[str, ...],
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return reciprocal product-based POX offspring.

    SOURCE_GAP: the papers do not fix how the job/product subset is sampled.  A
    non-empty random subset of RIACRSP products is retained from each fixed
    parent and the remaining positions are filled in the other parent's order.
    """

    if set(left) != set(instance.operations) or set(right) != set(instance.operations):
        raise ValueError("POX parents must be permutations of all operations")
    products = list(instance.products)
    if not products:
        raise ValueError("POX requires at least one product")
    if len(products) == 1:
        return left, right
    rng.shuffle(products)
    retained_count = rng.randrange(1, len(products)) if len(products) > 1 else 1
    retained = set(products[:retained_count])

    def child(
        fixed_parent: tuple[str, ...],
        fill_parent: tuple[str, ...],
        fixed_products: set[str],
    ) -> tuple[str, ...]:
        result: list[str | None] = [None] * len(fixed_parent)
        for index, operation in enumerate(fixed_parent):
            if instance.product_of[operation] in fixed_products:
                result[index] = operation
        fill = iter(
            operation
            for operation in fill_parent
            if instance.product_of[operation] not in fixed_products
        )
        return tuple(next(fill) if operation is None else operation for operation in result)

    complement = set(instance.products) - retained
    return child(left, right, retained), child(right, left, complement)


def uniform_pair(
    left: tuple[Gene, ...],
    right: tuple[Gene, ...],
    rng: random.Random,
) -> tuple[tuple[Gene, ...], tuple[Gene, ...]]:
    """Return reciprocal uniform-crossover offspring using one shared mask."""

    if len(left) != len(right):
        raise ValueError("uniform-crossover layers must have equal lengths")
    mask = [rng.random() < 0.5 for _ in left]
    return (
        tuple(a if keep_left else b for a, b, keep_left in zip(left, right, mask)),
        tuple(b if keep_left else a for a, b, keep_left in zip(left, right, mask)),
    )

from opmodel import (
    DType,
    EnergyBreakdown,
    EngineKind,
    GlobalFootprint,
    LocalOp,
    MemoryAccess,
    OpKind,
    OpProfile,
    Phase,
    TensorRole,
    TensorSpec,
)


def test_api_construction() -> None:
    op = LocalOp(
        name="x",
        kind=OpKind.ELEMENTWISE,
        phase=Phase.TRAIN_FWD,
        tensors=(TensorSpec(TensorRole.INPUT, (2, 3), DType.BF16),),
    )
    profile = OpProfile(
        latency_s=1.0,
        energy_j=2.0,
        flops=3.0,
        engine=EngineKind.VECTOR,
        footprint=GlobalFootprint(input_bytes=2, output_bytes=3),
        memory_access=MemoryAccess(hbm_read_bytes=2, hbm_write_bytes=3),
        energy_breakdown=EnergyBreakdown(compute_j=1.0, hbm_j=1.0),
        implementation="test",
    )
    assert op.name == "x"
    assert profile.footprint.total_bytes == 5
    assert profile.energy_breakdown.total_j == 2.0

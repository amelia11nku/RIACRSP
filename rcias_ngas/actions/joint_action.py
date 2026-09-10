from dataclasses import asdict, dataclass

from rcias_clgri.search.alns import REPAIR
from rcias_ngas.bank.provenance import Target
from rcias_ngas.evaluation.bks import content_hash
from .destroy_size import SIZE_FRACTIONS


@dataclass(frozen=True)
class JointAction:
    size: str
    target: Target
    repair: str

    def __post_init__(self):
        if self.size not in SIZE_FRACTIONS or self.repair not in REPAIR:
            raise ValueError('Invalid joint action component')
        if not self.target.operations:
            raise ValueError('Empty destroy target')

    @property
    def action_id(self):
        return 'ngas_action_' + content_hash(['ngas-action-v1', self.size, self.target.target_id, self.repair])[:24]

    def metadata(self):
        return {'action_id': self.action_id, 'size': self.size, 'repair': self.repair,
                'destroy_count': len(self.target.operations), 'target': asdict(self.target)}

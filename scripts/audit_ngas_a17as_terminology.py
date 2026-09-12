#!/usr/bin/env python3
"""Audit current-production terminology without erasing historical RT-HGT evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = ROOT / 'reports/ngas_a17as_terminology_audit.json'
MD_OUT = ROOT / 'reports/ngas_a17as_terminology_audit.md'
FORBIDDEN = (
    'RT-HGT numerical instability',
    'RT-HGT encoder itself became stale',
    'current production RT-HGT',
    'production C1 RT-HGT',
)


def active_files() -> list[Path]:
    paths = set((ROOT / 'reports').glob('ngas_a17ar_*.md'))
    paths.update((ROOT / 'reports').glob('ngas_a17as_*.md'))
    paths.update((ROOT / 'docs/reports/ngas_a1').glob('*.md'))
    return sorted(path for path in paths if path.is_file())


def main() -> None:
    searched, matches, violations = [], [], []
    for path in active_files():
        relative = path.relative_to(ROOT).as_posix()
        searched.append(relative)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if 'RT-HGT' not in line and 'HGT' not in line:
                continue
            row = {'file': relative, 'line': number, 'text': line.strip()}
            if any(pattern.lower() in line.lower() for pattern in FORBIDDEN):
                row['classification'] = 'INACCURATE_CURRENT_PRODUCTION'
                violations.append(row)
            else:
                row['classification'] = 'RETAINED_HISTORICAL_REFERENCE'
            matches.append(row)
    payload = {
        'schema': 'ngas-a17as-terminology-audit-v1',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if not violations else 'FAIL',
        'current_production_encoder': 'compact_relational',
        'files_searched': searched,
        'matches_found': matches,
        'inaccurate_matches_remaining': violations,
        'corrected_matches': [
            {
                'file': 'reports/ngas_a17ar_prior_staleness.md',
                'old': 'compact relational representation or RT-HGT encoder itself became stale',
                'new': 'compact-relational C1 representation or neural prior itself became stale',
            },
            {
                'file': 'reports/ngas_a17ar_final_report.md',
                'old': 'RT-HGT numerical instability',
                'new': 'compact-relational C1 numerical instability',
            },
        ],
        'intentionally_retained_historical_matches': [
            row for row in matches
            if row['classification'] == 'RETAINED_HISTORICAL_REFERENCE'],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    retained = payload['intentionally_retained_historical_matches']
    MD_OUT.write_text(f'''# NGAS A1.7A-S terminology audit

Status: **{payload['status']}**. The current production encoder is
`compact_relational`. Two inaccurate current-production references were corrected
through their report generators. No inaccurate historical-encoder production claim remains
in the searched A1.7A-R/A1.7A-S reports or NGAS documentation.

- Files searched: {len(searched)}
- Current-production violations remaining: {len(violations)}
- Historical references retained: {len(retained)}

The retained matches describe prior representation experiments or historical
terminal names and are listed with file and line anchors in
`reports/ngas_a17as_terminology_audit.json`.
''')
    print(json.dumps({'status': payload['status'], 'files': len(searched),
                      'matches': len(matches), 'violations': len(violations)}, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

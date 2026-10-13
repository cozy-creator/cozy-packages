#!/usr/bin/env python3
"""Read-only default-home record census; never invoke Cozy or a machine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def database(path):
    if not path.exists(): return {'path': str(path), 'exists': False}
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result = {'path': str(path), 'exists': True, 'tables': {}}
        for table in sorted(names):
            if table not in ('requests', 'executions') and not any(word in table for word in ('upload', 'export')):
                continue
            columns = [row[1] for row in db.execute('PRAGMA table_info("' + table.replace('"', '""') + '")')]
            selected = [name for name in ('id', 'request_id', 'execution_id', 'output_id', 'state', 'status', 'ordinal', 'namespace') if name in columns]
            if not selected: continue
            query = 'SELECT ' + ','.join('"' + name + '"' for name in selected) + ' FROM "' + table.replace('"', '""') + '"'
            rows = [dict(zip(selected, row)) for row in db.execute(query)]
            counts = {}
            for row in rows:
                state = str(row.get('state', row.get('status', 'unclassified')))
                counts[state] = counts.get(state, 0) + 1
            # IDs/states only: no prompt, credential, grant or signed URL is exported.
            result['tables'][table] = {'count': len(rows), 'states': counts, 'rows': rows}
        return result
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    if args.out.exists(): p.error('preserve prior census, use a fresh output path')
    result = {'observed_at': datetime.now(timezone.utc).isoformat(), 'read_only': True,
              'creator': database(Path.home() / '.cozy/creator.sqlite'),
              'runtime': database(Path.home() / '.tensorfs/.cozy-workspace/journal.sqlite3')}
    if args.baseline:
        before = json.loads(args.baseline.read_text()); changes = []
        for system in ('creator', 'runtime'):
            for table, old in before[system].get('tables', {}).items():
                current = result[system].get('tables', {}).get(table, {}).get('rows', [])
                for row in old['rows']:
                    protected = row.get('state', row.get('status')) == 'paused' or any(word in table for word in ('upload', 'export'))
                    if protected and row not in current: changes.append({'system': system, 'table': table, 'before': row})
        result['protected_rows_changed_or_missing'] = changes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as out: json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps({system: {table: data['states'] for table, data in result[system].get('tables', {}).items()} for system in ('creator', 'runtime')}, indent=2))
    if result.get('protected_rows_changed_or_missing'): raise SystemExit('protected history changed; diagnose, do not repair/delete automatically')


if __name__ == '__main__': main()

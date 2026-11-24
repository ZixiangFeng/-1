"""Server-owned assignment and common text policy for the CAT pilot."""
import secrets

TEXT_POLICY = 'shared-companion-v1'
DESIGNS = {'random': ('T', 'G', 'M'), 'random4': ('T', 'G', 'M', 'D')}


def init(c):
    c.execute('CREATE TABLE IF NOT EXISTS study_assignments(user TEXT PRIMARY KEY,design TEXT NOT NULL,method TEXT NOT NULL)')


def assign(c, requested, is_test):
    """Call inside BEGIN IMMEDIATE; existing participants are never reassigned.

    Sequential balance: uniformly choose among the currently smallest arms.
    This is balanced allocation, not independent four-way randomization.
    """
    if requested not in DESIGNS:
        if requested not in ('T', 'G', 'M', 'D', 'memory', 'control'):
            raise ValueError('分组无效')
        return requested
    arms = DESIGNS[requested]
    if requested == 'random4':
        counts = dict(c.execute('''SELECT u.arm,count(*) FROM users u JOIN study_assignments a ON a.user=u.id
            WHERE u.is_test=? AND a.design='cat4' GROUP BY u.arm''', (is_test,)))
    else:
        counts = dict(c.execute('SELECT arm,count(*) FROM users WHERE is_test=? GROUP BY arm', (is_test,)))
        counts['M'] = counts.get('M', 0) + counts.get('memory', 0)
    lowest = min(counts.get(arm, 0) for arm in arms)
    return secrets.choice([arm for arm in arms if counts.get(arm, 0) == lowest])

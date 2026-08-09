import re, subprocess, os, sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lessons')
sections = ['第06节-6.1-INTRODUCTION-Information-theory-is-found',
            '第06节-6.2-ENTROPY-Entropy-in-the-context-of-infor',
            '第06节-6.3-JOINT-AND-CONDITIONAL-ENTROPY',
            '第06节-6.3.1-Joint-Entropy-When-we-talk-about-two',
            '第06节-6.3.2-Conditional-Entropy-Conditional-entr',
            '第06节-6.4-INFORMATION-GAIN-Information-gain-qua',
            '第06节-6.5-MUTUAL-INFORMATION-Mutual-information',
            '第06节-6.6-DATA-COMPRESSION']

for s in sections:
    txt = open(os.path.join(base, s, '代码.md'), encoding='utf-8').read()
    m = re.search(r'```python\n(.*?)\n```', txt, re.S)
    code = m.group(1)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    actual = r.stdout
    mt = re.search(r'```text\n(.*?)\n```', txt, re.S)
    expected = mt.group(1)
    alines = [l for l in actual.splitlines() if l.strip()]
    elines = [l for l in expected.splitlines() if l.strip()]
    match = True
    diffs = []
    for i in range(max(len(alines), len(elines))):
        a = alines[i] if i < len(alines) else '<MISSING>'
        e = elines[i] if i < len(elines) else '<EXTRA>'
        ac = a.split('#')[0].strip()
        ec = e.split('#')[0].strip()
        if ac != ec:
            match = False
            diffs.append(f'  act[{i}]={a!r}')
            diffs.append(f'  exp[{i}]={e!r}')
    print(f'{s[:22]:<24} out_lines={len(alines)} vs txt_lines={len(elines)}  MATCH={match}')
    if not match:
        for d in diffs[:10]:
            print(d)

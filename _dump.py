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
    print('='*20, s[:20])
    print(r.stdout)

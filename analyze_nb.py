import json

with open(r'notebooks\TSAC_Group_Project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells: {len(cells)}")
print("=" * 80)

for i, c in enumerate(cells):
    ct = c['cell_type']
    src = ''.join(c['source'])
    # Show first 200 chars of source
    preview = src[:200].replace('\n', ' | ')
    print(f"\nCell {i} [{ct}]:")
    print(f"  {preview}")
    if ct == 'code' and c.get('outputs'):
        for out in c['outputs']:
            if out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))[:150]
                print(f"  OUTPUT: {text[:150]}")
            elif out.get('output_type') == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = ''.join(data['text/plain'])[:150]
                    print(f"  RESULT: {text[:150]}")

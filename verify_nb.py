import json

with open(r'notebooks\TSAC_Group_Project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Cells: {len(nb['cells'])}")
print(f"Format: nbformat {nb['nbformat']}.{nb['nbformat_minor']}")

md_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_count = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
print(f"Markdown: {md_count}, Code: {code_count}")

print("\nCell overview:")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = src.split('\n')[0][:80]
    print(f"  [{i:2d}] {c['cell_type']:8s} | {first_line}")

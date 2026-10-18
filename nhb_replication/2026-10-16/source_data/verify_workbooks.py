"""Independently reconcile exported Excel cell values against typed source inputs.

Uses only ZIP/XML readers; it does not create or modify Excel workbooks.
"""
from pathlib import Path
from math import isclose
import csv
import hashlib
import json
import posixpath
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
INPUT=json.loads((HERE/'workbook_inputs.json').read_text())
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
RNS='{http://schemas.openxmlformats.org/package/2006/relationships}'

def col(n):
    r=''
    while n:
        n,b=divmod(n-1,26);r=chr(65+b)+r
    return r

def strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist():return []
    s=ET.fromstring(z.read('xl/sharedStrings.xml'))
    return [''.join(el.itertext()) for el in s.findall('m:si',NS)]

def read_sheet(z,p,ss):
    root=ET.fromstring(z.read(p));cells={}
    for cell in root.findall('.//m:sheetData/m:row/m:c',NS):
        typ=cell.attrib.get('t');value=cell.find('m:v',NS)
        if typ=='s':v=ss[int(value.text)]
        elif typ=='inlineStr':v=''.join(cell.find('m:is',NS).itertext())
        elif value is None or value.text is None:v=None
        elif typ in ['str','e']:v=value.text
        elif typ=='b':v=value.text=='1'
        else:v=float(value.text)
        cells[cell.attrib['r']]=v
    pane=root.find('m:sheetViews/m:sheetView/m:pane',NS)
    tables=root.findall('m:tableParts/m:tablePart',NS)
    return cells,pane,len(tables)

reports=[];index=[];numeric=0;text=0;blank=0
selection=set(sys.argv[1:])
for book in INPUT['workbooks']:
    if selection and book['id'] not in selection:continue
    path=HERE/'outputs'/'nhb_20261016'/book['file'];assert path.exists(),path
    with ZipFile(path) as z:
        ss=strings(z)
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        relations={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels')).findall(RNS+'Relationship')}
        paths={s.attrib['name']:relations[s.attrib['{'+NS['r']+'}id']] for s in wb.findall('m:sheets/m:sheet',NS)}
        paths={name:posixpath.normpath(p.lstrip('/') if p.startswith('/') else 'xl/'+p) for name,p in paths.items()}
        assert list(paths)==['Readme']+[s['name'] for s in book['sheets']]
        report={'id':book['id'],'file':book['file'],'sheets':[],'value_mismatches':0}
        for s in book['sheets']:
            cells,pane,tables=read_sheet(z,paths[s['name']],ss)
            assert pane is not None and float(pane.attrib.get('ySplit',0))==5,(book['id'],s['name'],'freeze')
            assert tables>=1,(book['id'],s['name'],'native filtered table')
            checked=0
            for j,h in enumerate(s['headers'],1):assert cells.get(col(j)+'5')==h
            for i,row in enumerate(s['rows'],6):
                for j,expected in enumerate(row,1):
                    address=col(j)+str(i);actual=cells.get(address)
                    if expected is None:
                        assert actual in [None,''],(book['id'],s['name'],address,expected,actual);blank+=1
                    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
                        assert isinstance(actual,(int,float)) and isclose(actual,expected,rel_tol=2e-14,abs_tol=2e-14),(book['id'],s['name'],address,expected,actual)
                        numeric+=1
                    else:
                        assert actual==expected,(book['id'],s['name'],address,expected,actual);text+=1
                    checked+=1
            report['sheets'].append({'name':s['name'],'rows':len(s['rows']),'columns':len(s['headers']),'cells_checked':checked,'frozen_header_rows':5,'native_tables':tables})
        for p in z.namelist():
            if p.startswith('xl/worksheets/sheet') and p.endswith('.xml'):
                assert b't="e"' not in z.read(p),(book['id'],p,'Excel error cell')
            if p.startswith('xl/tables/') and p.endswith('.xml'):
                table=ET.fromstring(z.read(p))
                names=[c.attrib['name'].casefold() for c in table.findall('m:tableColumns/m:tableColumn',NS)]
                assert len(names)==len(set(names)),(book['id'],p,'Case-insensitive duplicate table headers')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    report.update(sha256=digest,bytes=path.stat().st_size)
    reports.append(report);index.append({'file':book['file'],'sha256':digest,'bytes':path.stat().st_size})

summary={'workbooks':len(reports),'numeric_cells_checked':numeric,'text_cells_checked':text,'blank_cells_checked':blank,'value_mismatches':0,'formula_error_cells':0,'all_source_values_reconciled':True,'native_table_headers_case_insensitive_unique':True,'frozen_headers_and_native_filtered_tables_verified':True,'reports':reports}
(HERE/('Partial_Workbook_Verification.json' if selection else 'Workbook_Verification.json')).write_text(json.dumps(summary,indent=2)+'\n')
with (HERE/('Partial_Workbook_SHA256SUMS.csv' if selection else 'Workbook_SHA256SUMS.csv')).open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['file','sha256','bytes']);writer.writeheader();writer.writerows(index)
print(json.dumps({k:v for k,v in summary.items() if k!='reports'},indent=2))

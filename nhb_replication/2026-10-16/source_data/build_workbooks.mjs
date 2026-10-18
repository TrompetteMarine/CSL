/** Figure source data. Authoring uses the pinned Codex artifact-tool runtime. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const root = process.argv[2];
if (!root) throw new Error('Usage: build_workbooks.mjs /absolute/path/to/source_data');
const input = JSON.parse(await fs.readFile(path.join(root, 'workbook_inputs.json'), 'utf8'));
const selectionArg=process.argv.slice(3).find(a=>!a.startsWith('--'));
const selection=selectionArg&&selectionArg!=='ALL'?selectionArg.split(','):null;
const finalise=process.argv.includes('--finalise');
const outputDir = path.join(root, 'outputs', 'nhb_20261016');
const qaDir = path.join(root, 'qa');
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const reports = [];
const colour = { header: '#23485F', ink: '#24313B', rule: '#C5D3DD', doc: '#84939D' };
const col = n => {
  let s=''; for(let v=n+1;v;v=Math.floor((v-1)/26))s=String.fromCharCode(65+(v-1)%26)+s; return s;
};

function baseStyle(sheet, cells, colourTab=null) {
  sheet.showGridLines=false;
  if(colourTab)sheet.tabColor=colourTab;
  const r=sheet.getRange(cells);
  r.format.font={ name:'Arial', size:10, color:colour.ink };
  r.format.verticalAlignment='center';
  r.format.rowHeight=17;
}

function addReadme(wb,spec,existing=false) {
  const sheet=existing?wb.worksheets.getItem('Readme'):wb.worksheets.add('Readme');
  baseStyle(sheet, `A1:B${spec.notes.length+5}`, colour.doc);
  sheet.getRange('A:A').format.columnWidth=28;
  sheet.getRange('B:B').format.columnWidth=112;
  sheet.getRange('A2').values=[[spec.title]];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:colour.ink};
  sheet.getRange('A3:B3').format.borders={bottom:{style:'thin',color:colour.rule}};
  sheet.getRange(`A5:B${spec.notes.length+4}`).values=spec.notes.map(x=>[x[0],x[1]]);
  sheet.getRange(`A5:A${spec.notes.length+4}`).format.font={name:'Arial',size:10,bold:true,color:colour.ink};
  sheet.getRange(`B5:B${spec.notes.length+4}`).format.wrapText=true;
  sheet.getRange(`A5:B${spec.notes.length+4}`).format.verticalAlignment='top';
  spec.notes.forEach((row,i)=>{sheet.getRange(`A${i+5}:B${i+5}`).format.rowHeight=Math.max(20,Math.ceil(row[1].length/110)*15+5);});
  return sheet;
}

function addData(wb,spec,t,i) {
  const sheet=wb.worksheets.add(t.name);
  const end=col(t.headers.length-1),rowCount=t.rows.length;
  const lastRow=rowCount+5;
  baseStyle(sheet,`A1:${end}${lastRow}`, i===0?colour.header:null);
  sheet.getRange(`A:${end}`).format.columnWidth=20;
  sheet.getRange('A2').values=[[t.title||t.name]];
  sheet.getRange('A2').format.font={name:'Arial',size:14,bold:true,color:colour.ink};
  sheet.getRange('A3').values=[[t.source||'Numerical source: see Readme.']];
  sheet.getRange('A3').format.font={name:'Arial',size:10,italic:true,color:'#61727E'};
  const header=sheet.getRange(`A5:${end}5`);
  header.values=[t.headers];
  header.format.fill=colour.header;
  header.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
  header.format.horizontalAlignment='center';
  header.format.wrapText=true;
  header.format.rowHeight=32;
  if(rowCount){
    // Block writes preserve numeric types and minimise remote-object overhead.
    const chunk=4000;
    for(let start=0;start<rowCount;start+=chunk){
      const rows=t.rows.slice(start,start+chunk);
      sheet.getRange(`A${start+6}:${end}${start+5+rows.length}`).values=rows;
    }
    sheet.getRange(`A6:${end}${lastRow}`).setNumberFormat('0.000000000');
    let bodyHeight=17;
    t.headers.forEach((h,j)=>{
      const letter=col(j);
      const values=t.rows.map(r=>r[j]).filter(v=>v!==null);
      const integer=values.length&&values.every(v=>typeof v==='number'&&Number.isInteger(v));
      const text=values.some(v=>typeof v==='string');
      if(text)sheet.getRange(`${letter}6:${letter}${lastRow}`).setNumberFormat('@');
      else if(integer)sheet.getRange(`${letter}6:${letter}${lastRow}`).setNumberFormat('#,##0');
      else if(values.some(v=>typeof v==='number'&&v!==0&&Math.abs(v)<1e-7))sheet.getRange(`${letter}6:${letter}${lastRow}`).setNumberFormat('0.000000000E+00');
      const longest=Math.max(h.length,...values.filter(v=>typeof v==='string').slice(0,500).map(v=>v.length));
      sheet.getRange(`${letter}:${letter}`).format.columnWidth=text?Math.min(44,Math.max(16,longest+2)):Math.max(17,Math.min(26,h.length+1));
      if(text&&longest>44){
        sheet.getRange(`${letter}6:${letter}${lastRow}`).format.wrapText=true;
        bodyHeight=Math.max(bodyHeight,32,Math.ceil(longest/44)*13);
      }
    });
    if(bodyHeight>17)sheet.getRange(`A6:${end}${lastRow}`).format.rowHeight=bodyHeight;
    const table=sheet.tables.add(`A5:${end}${lastRow}`,true,`Data_${spec.id.replace(/[^a-zA-Z0-9]/g,'')}_${i}`);
    table.showFilterButton=true;
  }
  sheet.freezePanes.freezeRows(5);
  if(t.headers.length>8)sheet.freezePanes.freezeColumns(Math.min(3,t.headers.length));
  return sheet;
}

for (const spec of input.workbooks) {
  if(selection&&!selection.includes(spec.id))continue;
  if(global.gc)global.gc();
  console.log(JSON.stringify({starting:spec.id,sheets:spec.sheets.length+1}));
  let wb;
  const affected=new Set(['Readme']);
  if(finalise){
    wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(outputDir,spec.file)));
    addReadme(wb,spec,true);
    spec.sheets.forEach((t,i)=>{
      const sheet=wb.worksheets.getItem(t.name),end=col(t.headers.length-1);
      const oldHeaders=sheet.getRange(`A5:${end}5`).values[0];
      if(JSON.stringify(oldHeaders)!==JSON.stringify(t.headers)){
        for(const table of [...sheet.tables.items])table.delete();
        sheet.getRange(`A5:${end}5`).values=[t.headers];
        const table=sheet.tables.add(`A5:${end}${t.rows.length+5}`,true,`Final_${spec.id}_${i}`);
        table.showFilterButton=true;
        affected.add(t.name);
      }
    });
  }else{
    wb=Workbook.create();
    addReadme(wb,spec);
    spec.sheets.forEach((t,i)=>{addData(wb,spec,t,i);console.log(JSON.stringify({populated:spec.id,sheet:t.name,rows:t.rows.length}));});
  }
  wb.recalculate();
  const report={id:spec.id,file:spec.file,sheets:[],formulaErrors:[]};
  const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:2500});
  report.formulaErrorsNDJSON=errors.ndjson;
  for(const name of ['Readme',...spec.sheets.map(t=>t.name)]){
    const idx=spec.sheets.findIndex(t=>t.name===name),t=idx>=0?spec.sheets[idx]:null;
    const columns=t?Math.min(t.headers.length,7):2;
    const end=col(columns-1);
    const last=t?Math.min(t.rows.length+5,17):spec.notes.length+4;
    if(finalise&&!affected.has(name)){
      report.sheets.push({name,rows:t.rows.length,columns:t.headers.length,unchangedRenderVerifiedPreviously:true});
      continue;
    }
    const inspection=await wb.inspect({kind:'table',range:`'${name}'!A1:${end}${Math.min(last,9)}`,include:'values,formulas',tableMaxRows:9,tableMaxCols:7,maxChars:2400});
    const preview=await wb.render({sheetName:name,range:`A1:${end}${last}`,scale:1.25,format:'png'});
    const png=`${spec.id}_${name.replace(/[^a-zA-Z0-9]/g,'_')}.png`;
    await fs.writeFile(path.join(qaDir,png),new Uint8Array(await preview.arrayBuffer()));
    report.sheets.push({name,rows:t?t.rows.length:spec.notes.length,columns:t?t.headers.length:2,preview:png,inspection:inspection.ndjson});
    if(t&&t.headers.length>7){
      const index=t.headers.findIndex(h=>['mean','effect','value','fraction','signal_sd','mean_conditional_correlation'].includes(h));
      if(index>=0){
        const start=col(index),finish=col(Math.min(t.headers.length-1,index+6));
        const numericPreview=await wb.render({sheetName:name,range:`${start}5:${finish}${last}`,scale:1.25,format:'png'});
        await fs.writeFile(path.join(qaDir,png.replace('.png','_numeric.png')),new Uint8Array(await numericPreview.arrayBuffer()));
      }
    }
  }
  const file=await SpreadsheetFile.exportXlsx(wb);
  const savePath=path.join(outputDir,spec.file+(finalise?'.finalising.xlsx':''));
  await file.save(savePath);
  // Export can emit a very large diagnostic snapshot. The scoped inspection
  // and independent cell audit below are the retained verification records.
  await fs.rm(savePath+'.inspect.ndjson',{force:true});
  if(finalise)await fs.rename(savePath,path.join(outputDir,spec.file));
  await fs.writeFile(path.join(qaDir,`${spec.id}_verification.json`),JSON.stringify(report,null,2)+'\n');
  reports.push({id:spec.id,file:spec.file,sheets:report.sheets.length,rows:spec.sheets.reduce((a,t)=>a+t.rows.length,0)});
  console.log(JSON.stringify(reports.at(-1)));
}
let combined=reports;
if(selection){
  const previous=JSON.parse(await fs.readFile(path.join(root,'Workbook_Build_Report.json'),'utf8'));
  combined=previous.filter(r=>!selection.includes(r.id)).concat(reports);
}
await fs.writeFile(path.join(root,'Workbook_Build_Report.json'),JSON.stringify(combined,null,2)+'\n');

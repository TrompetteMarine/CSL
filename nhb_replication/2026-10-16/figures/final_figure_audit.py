"""Physical-size, clipping, font embedding and rendered-text audit."""
from pathlib import Path
import json,hashlib,re
import fitz
P=Path(__file__).resolve().parent
names=['main01_concept','main06_population_limits','main02_two_communities','main03_size_geometry','main04_private_withdrawal','main05_many_communities','supp06_two_initialisations','supp01_graph_diagnostics','supp03_expanded_geometry','supp04_choice_degree','supp05_scaling','supp02_controls','supp07_many_detail','supp08_score_response','supp09_signal_anatomy','supp10_noise_propagation']
rows=[]
for name in names:
 f=P/(name+'.pdf');doc=fitz.open(f);page=doc[0]
 spans=[s for b in page.get_text('dict')['blocks'] if b.get('type')==0 for line in b['lines'] for s in line['spans']]
 outside=[{'text':s['text'],'bbox':s['bbox']} for s in spans if not (page.rect+(-.1,-.1,.1,.1)).contains(fitz.Rect(s['bbox']))]
 fonts=[]
 for font in page.get_fonts(full=True):
  info=doc.extract_font(font[0]);fonts.append({'name':font[3],'type':font[2],'embedded_bytes':len(info[3])})
 images=[]
 for item in page.get_images(full=True):
  for rect in page.get_image_rects(item[0]):
   images.append({'width_px':item[2],'height_px':item[3],'dpi_x':item[2]/rect.width*72,'dpi_y':item[3]/rect.height*72})
 svg=P/(name+'.svg')
 standard=[s for s in spans if 'MathExtension' not in s['font']]
 rows.append({'file':f.name,'pages':len(doc),'width_mm':page.rect.width/72*25.4,'height_mm':page.rect.height/72*25.4,'text_spans_outside_page':outside,'maximum_standard_text_pt':max(s['size'] for s in standard),'maximum_glyph_font_pt':max(s['size'] for s in spans),'minimum_body_text_pt':min(s['size'] for s in spans if len(s['text'].strip())>=4),'fonts':fonts,'raster_elements':images,'svg_editable_text':svg.exists() and '<text' in svg.read_text(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 assert len(doc)==1 and rows[-1]['width_mm']<=180.01 and rows[-1]['height_mm']<=170 and not outside,rows[-1]
 assert all(x['embedded_bytes'] for x in fonts),(name,fonts)
 assert rows[-1]['maximum_standard_text_pt']<=7.02,(name,rows[-1]['maximum_standard_text_pt'])
collision=json.loads((P/'Text_Collision_Audit.json').read_text())
assert len(collision)==13 and all(not x['text_collision_candidates'] for x in collision),collision
report={'published_figures':rows,'matplotlib_components_checked':len(collision),'text_collision_candidates':0,'standard_text_target_pt':'6.5–7; mathematical scripts are naturally smaller and extensible mathematical operators use their natural glyph font size','all_published_widths_at_most_180mm':True,'all_published_heights_at_most_170mm':True,'all_fonts_embedded':True,'all_text_inside_page':True,'guideline_urls':['https://www.nature.com/nathumbehav/submission-guidelines/aip-and-formatting','https://www.nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf'],'notes':['Automatic text-pair checks use the actually drawn tick labels and text-only bounding boxes, not annotation leader-arrow boxes.','Leaders, plotted marks and TikZ layout are additionally checked in rendered previews.','Only dense correlation-matrix artwork and colour gradients are raster elements; scientific labels, curves, uncertainty marks and diagram elements remain vector.']}
(P/'Figure_QA.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'published_figures':len(rows),'text_collisions':0,'clipped_spans':sum(len(x['text_spans_outside_page']) for x in rows),'max_width_mm':max(x['width_mm'] for x in rows),'max_height_mm':max(x['height_mm'] for x in rows),'max_standard_font_pt':max(x['maximum_standard_text_pt'] for x in rows)}))

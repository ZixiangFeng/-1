"""Index a locally trained/validated CLIP reproduction for the admin lab only."""
import argparse,base64,io,json,os,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import style_lab as lab
from semantic_training import sha

def run(root,data,user=''):
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    root=Path(root);data=Path(data);directory=root/'models/sticker-encoder'
    info=lab.reproduction_info(root)
    if not info['available']:raise ValueError('No trained reproduction checkpoint. Random/untrained weights are not allowed.')
    report=json.loads((directory/'reproduction.json').read_text())
    for name,digest in report['files'].items():
        path=(directory/name).resolve()
        if path.parent!=directory.resolve() or sha(path)!=digest:raise ValueError('Checkpoint file integrity failure')
    torch.set_num_threads(1)
    try:os.nice(10)
    except OSError:pass
    model=CLIPModel.from_pretrained(directory,local_files_only=True,use_safetensors=True).eval()
    processor=CLIPProcessor.from_pretrained(directory,local_files_only=True)
    c=sqlite3.connect('file:'+str((data/'study.sqlite3').resolve())+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
    rows=lab.catalog(c,user);c.close()
    for row in rows:
        with Image.open(io.BytesIO(base64.b64decode(row['image']))) as im:
            im=im.convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im)
            inputs=processor(images=bg.convert('RGB'),return_tensors='pt')
        with torch.inference_mode():
            v=torch.nn.functional.normalize(model.get_image_features(**inputs),dim=-1)[0]
        if not torch.isfinite(v).all():raise ValueError('Nonfinite embedding')
        with lab.cache(data) as db:
            db.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?)',(row['hash'],info['id'],json.dumps(v.tolist()),lab.utc()))
    print(json.dumps({'indexed':len(rows),'model':info['id'],'threshold':info['threshold'],'author_original':False}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--user',default='')
    a=p.parse_args();run(a.root,a.data,a.user)

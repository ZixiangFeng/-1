"""Validate configured text, JSON and image calls with synthetic data; optionally enable.
Usage: .venv/bin/python scripts/check_cloud.py [--enable]
Does not print credentials or participant records.
"""
import base64
import io
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cloud_api
from PIL import Image

path = Path(__file__).resolve().parents[1] / 'data/ark-config.json'
config = json.loads(path.read_text())
try:
    if not config.get('vision_model'):
        raise ValueError('缺少 vision_model：请填写已开通的图片理解模型 ID，可与 model 相同')
    cloud_api.generate(config,[{'role':'user','content':'你好，请简短回答。'}])
    schema={'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']}
    text,_=cloud_api.generate(config,[{'role':'user','content':'返回 JSON：ok 为 true。'}],schema=schema,planning=True)
    if json.loads(text) != {'ok':True}:
        raise ValueError('结构化输出验证失败')
    buf=io.BytesIO()
    Image.new('RGB',(256,256),(255,0,0)).save(buf,format='JPEG')
    answer,_=cloud_api.generate(config,[{'role':'user','content':'这张图主要是什么颜色？只回答中文颜色名。','images':[base64.b64encode(buf.getvalue()).decode()]}],vision=True)
    if '红' not in answer:
        raise ValueError('图片颜色识别检查未通过')
    if '--enable' in sys.argv:
        config['enabled']=True
        temp=path.with_suffix('.tmp')
        temp.touch(mode=0o600,exist_ok=True)
        temp.chmod(0o600)
        temp.write_text(json.dumps(config,ensure_ascii=False,indent=2))
        temp.replace(path)
    print('文字、JSON、图片三项真实接口检查通过。'+('已启用云端。' if '--enable' in sys.argv else '配置未改变。'))
except Exception as e:
    print(str(e) if isinstance(e,ValueError) else '接口检查失败；配置未启用。')
    sys.exit(1)

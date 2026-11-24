"""Admin-only screenshot crops for G/D libraries, never participant memories."""
import base64
import io
import json
import secrets
from datetime import datetime, timezone, timedelta
from PIL import Image
import screenshot_library as crops
import generic_stickers


def init(c):
    c.execute('''CREATE TABLE IF NOT EXISTS generic_screenshot_drafts(
        id TEXT PRIMARY KEY,pool TEXT NOT NULL,style TEXT NOT NULL,name TEXT NOT NULL,
        image BLOB NOT NULL,boxes TEXT NOT NULL,created TEXT NOT NULL)''')


def preview(c, data):
    pool=data.get('pool','generic');style=str(data.get('style','')).strip()
    generic_stickers.validate_target(pool,style)
    image=crops.read_image(data.get('image',''))
    method,boxes=crops.detect(image,data.get('options'))
    out=io.BytesIO();image.save(out,'PNG')
    draft=secrets.token_hex(16);name=str(data.get('name','截图'))[:120]
    c.execute('DELETE FROM generic_screenshot_drafts WHERE created<?',
              ((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),))
    c.execute('INSERT INTO generic_screenshot_drafts VALUES(?,?,?,?,?,?,?)',
              (draft,pool,style,name,out.getvalue(),json.dumps(boxes),crops.now()))
    return {'draft':draft,'pool':pool,'style':style,'name':name,'width':image.width,'height':image.height,'method':method,
            'tiles':[{**b,'image':base64.b64encode(crops.jpeg(image.crop(b['box']),240)).decode()} for b in boxes]}


def import_tile(c, data, generate, event):
    # The target is frozen in the preview; a later dropdown change cannot route
    # a pending crop into a different experimental library.
    draft=data.get('draft');index=data.get('index')
    if not isinstance(draft,str) or type(index) is not int:
        raise ValueError('裁切预览无效，请重新选择截图')
    record=c.execute('SELECT * FROM generic_screenshot_drafts WHERE id=?',(draft,)).fetchone()
    if not record or record['created']<(datetime.now(timezone.utc)-timedelta(days=1)).isoformat():
        raise ValueError('裁切预览已过期，请重新裁切；已入库图片不会丢失')
    boxes={b['index']:b for b in json.loads(record['boxes'])}
    if index not in boxes:raise ValueError('裁切编号无效')
    with Image.open(io.BytesIO(record['image'])) as image:
        box=crops.box_checked(data.get('box',boxes[index]['box']),image.size)
        binary=crops.jpeg(image.crop(box))
    result=generic_stickers.store_prepared(c,binary,base64.b64encode(binary).decode(),record['pool'],record['style'],generate,event)
    if not result['duplicate']:
        event(c,'__generic__','generic_screenshot_import',{'asset':result['id'],'pool':record['pool'],
            'style':record['style'],'draft':draft,'tile':index,'box':box,'source_name':record['name']})
    return dict(result,pool=record['pool'],index=index)

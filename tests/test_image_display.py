import base64,io,os,tempfile,unittest
os.environ.setdefault('DEMO_DATA',tempfile.mkdtemp())
from PIL import Image
from server import display_image

class DisplayImages(unittest.TestCase):
    def test_large_image_is_smaller_but_original_is_retained(self):
        image=Image.effect_noise((768,768),90).convert('RGB')
        out=io.BytesIO();image.save(out,format='JPEG',quality=85)
        original=out.getvalue();encoded=base64.b64encode(original).decode()
        result=display_image(encoded)
        self.assertLess(len(result),len(original));self.assertLessEqual(len(result),80000)
        decoded=Image.open(io.BytesIO(result));decoded.load()
        self.assertLessEqual(max(decoded.size),480)
        self.assertEqual(base64.b64decode(encoded),original)
        print('Display compression fixture:',len(original),'->',len(result),'bytes')
    def test_small_image_never_upscaled_or_made_larger(self):
        out=io.BytesIO();Image.new('RGB',(96,64),'green').save(out,format='JPEG',quality=75,optimize=True)
        original=out.getvalue();result=display_image(base64.b64encode(original).decode())
        self.assertLessEqual(len(result),len(original));self.assertEqual(Image.open(io.BytesIO(result)).size,(96,64))

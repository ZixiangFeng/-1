import unittest
import torch
from research.sticker_encoder.local_pairs import image_pair_loss,validate_pairs
class LocalPairTests(unittest.TestCase):
 def test_holdouts_and_human_provenance_cannot_be_used(self):
  rows=[{'id':'a','split':'train','quality':'clear'},{'id':'b','split':'test','quality':'clear'}]
  p={'left':'a','right':'b','label':1,'reviewer':'codex_ai','human_reviewed':False}
  with self.assertRaises(ValueError):validate_pairs([p],rows)
  rows[1]['split']='train';p['human_reviewed']=True
  with self.assertRaises(ValueError):validate_pairs([p],rows)
  p['human_reviewed']=False;validate_pairs([p],rows)
 def test_loss_direction_separates_negative_and_aligns_positive(self):
  a=torch.tensor([[1.,0.]],requires_grad=True);same=torch.tensor([[1.,0.]]);opposite=torch.tensor([[0.,1.]])
  self.assertLess(image_pair_loss(a,same,torch.tensor([1])).item(),image_pair_loss(a,opposite,torch.tensor([1])).item())
  self.assertGreater(image_pair_loss(a,same,torch.tensor([0])).item(),image_pair_loss(a,opposite,torch.tensor([0])).item())
 def test_uncertain_and_duplicate_pairs_rejected(self):
  rows=[{'id':'a','split':'train','quality':'clear'},{'id':'b','split':'train','quality':'uncertain'}]
  p={'left':'a','right':'b','label':1,'reviewer':'codex_ai','human_reviewed':False}
  with self.assertRaises(ValueError):validate_pairs([p],rows)
  rows[1]['quality']='clear'
  with self.assertRaises(ValueError):validate_pairs([p,p],rows)
if __name__=='__main__':unittest.main()

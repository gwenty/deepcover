#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
import pandas as pd
from PIL import Image
import copy
import sys, os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras.applications import inception_v3, mobilenet, xception
from torch import sigmoid
import sys
from torch import tensor

class explain_objectt:
  def __init__(self, model, inputs):
    self.model=model
    self.inputs=inputs
    self.outputs=None
    self.top_classes=None
    self.adv_ub=None
    self.adv_lb=None
    self.adv_value=None
    self.testgen_factor=None
    self.testgen_size=None
    self.testgen_iter=None
    self.vgg16=None
    self.mnist=None
    self.cifar10=None
    self.inception_v3=None
    self.xception=None
    self.mobilenet=None
    self.attack=None
    self.text_only=None
    self.measures=None
    self.normalized=None
    self.x_verbosity=None
    self.fnames=[]
    self.boxes=None
    self.occlusion_file=None
    self.min_exp=1.1
    self.immunobert=False
    self.immunobert_path=None
    self.convert_example_to_batch=None
    self.move_dict_to_device=None
  
  
  # I want to call the same prediction function regerdless of model
  def predict(self, x):
    if self.immunobert:
      res=float(sigmoid(self.model(self.move_dict_to_device(self.convert_example_to_batch((make_instance(x))), self.model))))
      #res = [[1-res, res]]
      #print(res)
      y = int(np.round(res)) # Setting the cutoff point to 0.5
    else:
      res=self.model.predict(sbfl_preprocess(self, np.array([x])))
      y=np.argsort(res)[0][-self.top_classes:]
    return y

  def set_immunobert(self, immunobert_path):
    self.immunobert=True
    self.immunobert_path = immunobert_path
    #self.adv_value = 21
    sys.path.append(self.immunobert_path + os.path.sep + 'epitope')
    from pMHC.data.utils import convert_example_to_batch, move_dict_to_device
    self.convert_example_to_batch = convert_example_to_batch
    self.move_dict_to_device = move_dict_to_device
    
def make_instance(arrays):
  return {'input_ids':tensor(arrays[0]), 'token_type_ids':tensor(arrays[1]), 'position_ids':tensor(arrays[2]),
          'input_mask':tensor(arrays[3]), 'targets':tensor(arrays[4])}

def get_peptide(x, token_type_id=2):
  # 1 is token_type_id - 2 is peptide
  # 0 is input_id - remove 2 and 3 (sep and cls)
  peptide_positions = (x[1] == token_type_id) & (x[0] != 2) & (x[0] != 3)
  peptide = x[0][peptide_positions]
  peptide_idx = np.where(peptide_positions)[0]
  return peptide, peptide_idx

def get_regions(x, token_type_ids=[2]):
  peptide_positions = (x[0] != 2) & (x[0] != 3)
  ok_token_ids = np.array([False]*len(x[0]))
  for token_type_id in token_type_ids:
    ok_token_ids = np.array(ok_token_ids) | (np.array(x[1] == token_type_id))
  peptide_positions = peptide_positions & ok_token_ids
  peptide = x[0][peptide_positions]
  peptide_idx = np.where(peptide_positions)[0]
  return peptide, peptide_idx

def insert_peptide(x, peptide, peptide_idx):
  x[0][peptide_idx] = peptide
  return x

class sbfl_elementt:
  def __init__(self, x, y, xs, ys, model, fname=None, adv_part=None, immunobert=False):
    self.x=x
    self.y=y
    self.xs=xs
    self.ys=ys
    self.model=model
    self.fname=fname
    self.adv_part=adv_part
    self.immunobert=immunobert

class substitution_matrix:
  def __init__(self, blosum_path):
    self.IUPAC_VOCAB = dict([
      ("<pad>", 0),
      ("<mask>", 1),
      ("<cls>", 2),
      ("<sep>", 3),
      ("<unk>", 4),
      ("*", 4),
      ("A", 5),
      ("B", 6),
      ("C", 7),
      ("D", 8),
      ("E", 9),
      ("F", 10),
      ("G", 11),
      ("H", 12),
      ("I", 13),
      ("K", 14),
      ("L", 15),
      ("M", 16),
      ("N", 17),
      ("O", 18),   # not in humans
      ("P", 19),
      ("Q", 20),
      ("R", 21),
      ("S", 22),
      ("T", 23),
      ("U", 24),
      ("V", 25),
      ("W", 26),
      ("X", 27),
      ("Y", 28),
      ("Z", 29)]) # 25

    # Initialise the substitution matrix
    self.blosum = pd.read_csv(blosum_path, sep='\s+')
    self.blosum.index = [self.IUPAC_VOCAB[a] for a in self.blosum.index]
    self.blosum.columns = [self.IUPAC_VOCAB[a] for a in self.blosum.columns]

# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def arr_to_str(inp):
  ret=inp[0]
  for i in range(1, len(inp)):
    ret+=' '
    ret+=inp[i]
  return ret

def sbfl_preprocess(eobj, chunk):
  x=chunk.copy()
  if eobj.vgg16 is True:
    x=vgg16.preprocess_input(x)
  elif eobj.inception_v3 is True:
    x=inception_v3.preprocess_input(x)
  elif eobj.xception is True:
    x=xception.preprocess_input(x)
  elif eobj.mobilenet is True:
    x=mobilenet.preprocess_input(x)
  elif eobj.normalized is True:
    x=x/255.
  elif eobj.mnist is True or eobj.cifar10 is True:
    x=x/255.
  return x

def save_an_image(im, title, di='./'):
  if not di.endswith('/'):
    di+='/'
  save_img((di+title+'.jpg'), im)

def save_spectrum(spectrum, filename):
  print(spectrum.shape)
  print(len(spectrum.shape)==3)
  if len(spectrum.shape) == 3:
    with open(filename,'w') as f:
      for a in spectrum:
          np.savetxt(f, a, fmt='%s')
          f.write('\n')
  elif len(spectrum.shape)==1 or len(spectrum.shape)==2:
    np.savetxt(filename, spectrum, fmt='%s')

def immunobert_spectrum_plot_dict(example, spectrum, filename):
  fig = plt.figure(figsize=(8, 6))

  nflank_ax = fig.add_subplot(3,3,1)
  peptide_ax = fig.add_subplot(3,3,2)
  cflank_ax = fig.add_subplot(3,3,3)
  mhc_ax = fig.add_subplot(3,1,2)

  print(spectrum)
  ymin = 0
  ymax = np.max(spectrum) * 1.1

  y = spectrum[0][example['token_type_ids'].numpy() == 1]
  nflank_ax.bar(np.arange(len(y)), y)
  nflank_ax.set_title('N-flank')
  nflank_ax.set_ylim([ymin, ymax])

  y = spectrum[0][example['token_type_ids'].numpy() == 2]
  peptide_ax.bar(np.arange(len(y)), y)
  peptide_ax.set_title('Peptide')
  peptide_ax.set_ylim([ymin, ymax])

  y = spectrum[0][example['token_type_ids'].numpy() == 3]
  cflank_ax.bar(np.arange(len(y)), y)
  cflank_ax.set_title('C-flank')
  cflank_ax.set_ylim([ymin, ymax])

  y = spectrum[0][example['token_type_ids'].numpy() == 0]
  mhc_ax.bar(np.arange(len(y)), y)
  mhc_ax.set_title('MHC allele')
  mhc_ax.set_ylim([ymin, ymax])

  fig.tight_layout()

  fig.savefig(filename)

def immunobert_spectrum_plot(example, spectrum, filename):
  fig = plt.figure(figsize=(8, 6))

  nflank_ax = fig.add_subplot(3,3,1)
  peptide_ax = fig.add_subplot(3,3,2)
  cflank_ax = fig.add_subplot(3,3,3)
  mhc_ax = fig.add_subplot(3,1,2)

  spectrum=np.nan_to_num(spectrum)
  #print(spectrum)
  if np.min(spectrum) < 0:
    ymin = np.min(spectrum) * 1.1
  else:
    ymin = 0
  ymax = np.max(spectrum) * 1.1

  y = spectrum[0][(example[1] == 1) & (example[0] != 2) & (example[0] != 3)]
  nflank_ax.bar(np.arange(len(y)), y)
  nflank_ax.set_title('N-flank')
  nflank_ax.set_ylim([ymin, ymax])

  y = spectrum[0][(example[1] == 2) & (example[0] != 2) & (example[0] != 3)]
  peptide_ax.bar(np.arange(len(y)), y)
  peptide_ax.set_title('Peptide')
  peptide_ax.set_ylim([ymin, ymax])

  y = spectrum[0][(example[1] == 3) & (example[0] != 2) & (example[0] != 3)]
  cflank_ax.bar(np.arange(len(y)), y)
  cflank_ax.set_title('C-flank')
  cflank_ax.set_ylim([ymin, ymax])

  y = spectrum[0][(example[1] == 0) & (example[0] != 2) & (example[0] != 3)]
  mhc_ax.bar(np.arange(len(y)), y)
  mhc_ax.set_title('MHC allele')
  mhc_ax.set_ylim([ymin, ymax])

  fig.tight_layout()

  fig.savefig(filename)

def top_plot(sbfl_element, ind, di, metric='', eobj=None, bg=128, online=False, online_mark=[255,0,255]):
  origin_data=sbfl_element.x
  sp=origin_data.shape

  try:
    #print ('mkdir -p {0}'.format(di))
    os.system('mkdir -p {0}'.format(di))
  except: pass

  save_an_image(origin_data, 'origin-{0}'.format(sbfl_element.y), di)

  ret=None

  im_flag=np.zeros(sp, dtype=bool)
  im_o=np.multiply(np.ones(sp), bg)
  count=0
  base=int((ind.size/sp[2])/100)
  pos=ind.size-1
  found_exp = False
  while pos>=0:
    ipos=np.unravel_index(ind[pos], sp)
    if not im_flag[ipos]:
      for k in range(0,sp[2]):
        im_o[ipos[0]][ipos[1]][k]=origin_data[ipos[0]][ipos[1]][k]
        im_flag[ipos[0]][ipos[1]][k]=True
      count+=1
      if count%base==0:
        save_an_image(im_o, '{1}-{0}'.format(int(count/base), metric), di)
        res=sbfl_element.model.predict(sbfl_preprocess(eobj, np.array([im_o])))
        y=np.argsort(res)[0][-eobj.top_classes:]
        #print (int(count/base), '>>>', y, sbfl_element.y, y==sbfl_element.y)
        if y==sbfl_element.y and not found_exp: 
          save_an_image(im_o, 'explanation-found-{1}-{0}'.format(int(count/base), metric), di)
          found_exp = True
          if not eobj.boxes is None: # wsol calculation
              vect=eobj.boxes[sbfl_element.fname.split('/')[-1]]
              ref_flag=np.zeros(sp, dtype=bool)
              ref_flag[vect[0]:vect[2], vect[1]:vect[3], :]=True

              union=np.logical_or(im_flag, ref_flag)
              inter=np.logical_and(im_flag, ref_flag)
              iou=np.count_nonzero(inter)*1./np.count_nonzero(union)
              ret=iou
          elif not eobj.occlusion_file is None: # occlusion calculation
                ref_flag=np.zeros(sp, dtype=bool)
                for i in range(0, sp[0]):
                    for j in range(0, sp[1]):
                        if origin_data[i][j][0] == 0 and origin_data[i][j][1] == 0 and origin_data[i][j][2] == 0:
                            ref_flag[i][j][:] = True

                union=np.logical_or(im_flag, ref_flag)
                inter=np.logical_and(im_flag, ref_flag)
                iou=np.count_nonzero(inter)*1./np.count_nonzero(union)
                intersection=np.count_nonzero(inter)*1./np.count_nonzero(ref_flag)
                ret=[(count/base)/100., intersection, iou]

          if eobj.x_verbosity>0: return ret

    pos-=1
  return ret


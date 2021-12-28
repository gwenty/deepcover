import numpy as np
from utils import *
from torch import tensor
import copy
from random import randint, choice, choices
from itertools import combinations

# This is where the mutations happen
def spectra_sym_gen(eobj, x, y, adv_value=1, testgen_factor=.2, testgen_size=0, substitution_matrix=None):
  
  v_type=type(adv_value)
  failing=[]
  passing=[]

  #inputs=[]
  if eobj.immunobert:
    #sp=[1,x['input_ids'].shape[0]] # It is 1D
    sp_mutate=[1,x[0].shape[0]]  # sp holds shape of to be mutated
    sp_x=[x.shape[0], x[0].shape[0]]
  else:
    sp=x.shape
  x_flag=np.zeros(sp, dtype=bool)
  portion=int(sp[0]*testgen_factor)
  incr=1/6*portion
  if portion<1: portion=1
  if eobj.immunobert:
    L0=np.array(np.arange(x['input_ids'].numel()))
  else:
    L0=np.array(np.arange(x.size))
  L0=np.reshape(L0, sp)

  bert_portion = 2
  
  # Until all are masked or set is big enough. 
  while (not np.all(x_flag)) or len(passing)+len(failing)<testgen_size:
    #print ('####', len(passing), len(failing))
    if eobj.immunobert:
      t=copy.deepcopy(x)
    else:
      t=x.copy()

    i0=np.random.randint(0,sp[0])
    i1=np.random.randint(0,sp[1])

    if eobj.immunobert:
      # Mutate only the peptide
      peptide = t['input_ids'][t['token_type_ids'] == 2]
      # Remove sep and cls 
      peptide = peptide[(peptide != 2) & (peptide != 3)]
      h=portion
      region=L0[ np.max([i0-h,0]) : np.min([i0+h, sp[0]]), np.max([i1-h,0]):np.min([i1+h,sp[1]])].flatten()
      L=region #L0[0:portion]
    else:
      h=portion
      region=L0[ np.max([i0-h,0]) : np.min([i0+h, sp[0]]), np.max([i1-h,0]):np.min([i1+h,sp[1]])].flatten()
      L=region #L0[0:portion]

    if v_type==np.ndarray:
      np.put(t, L, adv_value.take(L))
    else:
      if eobj.immunobert:
        t['input_ids'][L] = tensor(substitute_aa(t['input_ids'][L].numpy(), substitution_matrix))
      else: 
        np.put(t, L, adv_value)
    x_flag.flat[L]=True #np.put(x, L, True)
    new_y=eobj.predict(t)
    is_adv=(len(np.intersect1d(y, new_y))==0)
    # print(y)
    # print(new_y)
    # print(is_adv)

    # predicted to not y
    if is_adv:
      failing.append(t)
      ## to find a passing
      ite=h #testgen_factor
      while ite>1: #ite>0.01:
        t2=x.copy()
        #ite=ite-1#ite//2 #ite=(ite+0)/2
        ite=int(ite-incr)
        if ite<1: break
        region=L0[ np.max([i0-ite,0]) : np.min([i0+ite, sp[0]]), np.max([i1-ite,0]):np.min([i1+ite,sp[1]])].flatten()

        L=region #L0[0:portion]
        if v_type==np.ndarray:
          np.put(t, L, adv_value.take(L))
        else:
          if eobj.immunobert:
            t['input_ids'][L] = tensor(substitute_aa(t['input_ids'][L].numpy(), substitution_matrix))
          else: 
            np.put(t, L, adv_value)
        x_flag.flat[L]=True #np.put(x, L, True)
        new_y=eobj.predict(t)
        #is_adv=(len(np.intersect1d(y, new_y))==0)
        #ite-=0.01
        #L2=L0[0:int(ite/testgen_factor*portion)]
        #if v_type==np.ndarray:
        #  np.put(t2, L2, adv_value.take(L2))
        #else:
        #  np.put(t2, L2, adv_value)
        #new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t2]))))[0][-eobj.top_classes:]
        ##print (y, new_y)
        if (len(np.intersect1d(y, new_y))!=0):
          passing.append(t2)
          break
    
    # predicted to y
    else:
      passing.append(t)
      ## to find a failing
      ite=h #testgen_factor   # = 1
      while ite<sp[0]/2: #0.99: # = 0.5
        t2=x.copy()
        #ite=ite+1#ite*2
        ite=int(ite+incr)
        if ite>sp[0]/2: break
        region=L0[ np.max([i0-ite,0]) : np.min([i0+ite, sp[0]]), np.max([i1-ite,0]):np.min([i1+ite,sp[1]])].flatten()

        L=region #L0[0:portion]
        if v_type==np.ndarray:
          np.put(t, L, adv_value.take(L))
        else:
          np.put(t, L, adv_value)
        x_flag.flat[L]=True #np.put(x, L, True)
        new_y=eobj.predict(t)
        #t2=x.copy()
        #ite=(ite+1)/2
        ##ite+=0.01
        #L2=L0[0:int(ite/testgen_factor*portion)]
        #if v_type==np.ndarray:
        #  np.put(t2, L2, adv_value.take(L2))
        #else:
        #  np.put(t2, L2, adv_value)
        #new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t2]))))[0][-eobj.top_classes:]
        if (len(np.intersect1d(y, new_y))==0):
          failing.append(t2)
          x_flag.flat[L]=True
          break

  return np.array(passing), np.array(failing)


def mutate(x):
  # Don't mutate <cls> or <sep> (input id = 0-3)
  # Single mutation site
  #randint(0,len(x))
  return

def substitute_aa(region, sub_mat):
  blosum = sub_mat.blosum
  new_region = []
  #print('sub_aa')
  #print(region)
  for aa in region:
    # Find a substitution for an AA
    max_sub_value = blosum[aa].drop(aa).max() # Most similar
    # If there are mutliple maxs, randomly pick one to sub with
    sub_aa = choice(blosum[aa][blosum[aa]==max_sub_value].index.values)
    new_region.append(sub_aa)
  #print(new_region)
  return new_region

def spectra_gen(x, adv_value=1, testgen_factor=0.01, testgen_size=0):

  #print (adv_value, testgen_factor, testgen_size)
  v_type=type(adv_value)

  inputs=[]
  sp=x.shape
  x_flag=np.zeros(sp, dtype=bool)
  portion=int(x.size*testgen_factor) #int(x.size/sp[2]*testgen_factor)
  
  while (not np.all(x_flag)) or len(inputs)<testgen_size:
    t=x.copy()
    L=np.random.choice(x.size, portion)
    if v_type==np.ndarray:
      #t.flat[L]=adv_value.take(L) 
      np.put(t, L, adv_value.take(L))
    else:
      #t.flat[L]=adv_value 
      np.put(t, L, adv_value)
    x_flag.flat[L]=True #np.put(x, L, True)
    #for pos in L:
    #  ipos=np.unravel_index(pos,sp) 
    #  #if v_type==np.ndarray:
    #  #  t.flat[pos]=adv_value.flat[pos]
    #  #else: t.flat[pos]=adv_value
    #  #x_flag.flat[pos]=True #np.put(x, L, True)
    #  for j in range(0, sp[2]):
    #    if v_type==np.ndarray:
    #      t[ipos[0]][ipos[1]][j]=adv_value[ipos[0]][ipos[1]][j]
    #    else:
    #      t[ipos[0]][ipos[1]][j]=adv_value
    #    x_flag[ipos[0]][ipos[1]][j]=True
    inputs.append([t])
  return inputs

def all_combos(x, n_to_mutate):
  # All combos 
  to_mask = list(combinations(range(len(x)),n_to_mutate))
  if n_to_mutate == 1:
      to_mask = [m[0] for m in to_mask]
  return to_mask

# Use this if not dynamic size
def spectra_gen_immunobert(eobj, x, y, substitution_matrix, mask_sizes=[1,2,3], regions_to_mask=[2, 0], testgen_size=-1):
# Regions to mask: The token type ids to mutate. 2=peptide. 1,3=flanks. 0=mhc prot.

  passing = []
  failing = []
  inputs = []

  # Mutate only peptide
  peptide, peptide_idx = get_regions(x, token_type_ids=regions_to_mask)

  for n_to_mutate in mask_sizes:

    # Get all index combinations of the specified sizes
    idx_to_mask = all_combos(peptide, n_to_mutate)
    print(idx_to_mask)
    # Choose testgen_size of these
    if testgen_size != -1 and len(idx_to_mask)>testgen_size:
      idx_to_mask = choices(idx_to_mask, k=testgen_size)

    # Mutate those peptide locations
    for region_index in idx_to_mask:
      # Get the AAs in those idxs
      aa_region = peptide[[region_index]]
      if n_to_mutate == 1:
          new_region = substitute_aa([aa_region], substitution_matrix)
      else: 
          new_region = substitute_aa(aa_region, substitution_matrix)
      peptide_copy = copy.deepcopy(peptide)
      peptide_copy[[region_index]] = new_region
      # Put new peptide back in the instance
      t = copy.deepcopy(x)
      t = insert_peptide(t, peptide_copy, peptide_idx)
      #print(eobj.predict(t))
      inputs.append(t)
      # Same as original - passing
      # Not same as original - failing

  new_ys = np.array([eobj.predict(inpu) for inpu in inputs])
  inputs = np.array(inputs)
  passing = inputs[new_ys == y]
  failing = inputs[new_ys != y]

  #print(peptide_idx)

  return passing, failing, peptide_idx

import numpy as np
from utils import *
import copy

def to_rank(sbfl_element, metric='zoltar'):
  if sbfl_element.immunobert:
    # This will be an array of arrays
    origin_data = sbfl_element.x[0] # Just the input array
    #print(origin_data)
    sp=[1,origin_data.shape[0]] # It is 1D
  else:
    origin_data=sbfl_element.x
    sp=origin_data.shape
  ef=np.zeros(sp,dtype=float)
  nf=np.zeros(sp,dtype=float)
  ep=np.zeros(sp,dtype=float)
  np_=np.zeros(sp,dtype=float)

  xs=np.array(sbfl_element.xs)

  # The difference between each generated mutant and the instance to be explained
  if sbfl_element.immunobert:
    diffs = copy.deepcopy(xs)
    for i, x in enumerate(xs):
      diffs[i][0] = x[0] - origin_data
  else:
    diffs=np.abs(xs-origin_data)
  #diffs=diffs - (1+0.05 * origin_data)
  #diffs[diffs>0]=0

  for i in range(0, len(diffs)): # TODO: Not sure what happens here
    is_adv=(sbfl_element.y!=sbfl_element.ys[i])
    if sbfl_element.immunobert:
      ds_i1 = np.array(copy.deepcopy(diffs[i])[0])
      ds_i2=np.array(copy.deepcopy(diffs[i])[0])
      
    else:
      ds_i1=diffs[i].copy()
      ds_i2=diffs[i].copy()

    ds_i1[ds_i1!=0]=1
    ds_i2[ds_i2!=0]=-1
    ds_i2[ds_i2==0]=+1 # When difference is 0, it was not masked (or originally the same value = same effect as not masking)
    ds_i2[ds_i2==-1]=0

    if is_adv: # predicted to not y
      ef=ef+ds_i1 # masked
      nf=nf+ds_i2 # not masked TODO: This makes no sense, I härled to the opposite as supposed to be by text.
      #ef=ef+ds_i2
      #nf=nf+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ef[index]+=1
      #  else:
      #    nf[index]+=1
    else:
      ep=ep+ds_i1
      np_=np_+ds_i2
      #ep=ep+ds_i2
      #np_=np_+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ep[index]+=1
      #  else:
      #    np_[index]+=1

  if np.min(ef) < 0:
    print('ef: {}'.format(ef))
  if np.min(nf) < 0:
    print('nf: {}'.format(nf))
  if np.min(ef) < 0:
    print('ep: {}'.format(ep))
  if np.min(np_) < 0:
    print('np_: {}'.format(np_))
  ind=None
  spectrum=None
  if sbfl_element.immunobert:
    origin_data2 = [origin_data]
  else:
    origin_data2 = origin_data
  if metric=='random':
    spectrum=np.random.rand(sp[0], sp[1], sp[2])
  elif metric=='zoltar':
    zoltar=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data2):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      if aef==0:
        zoltar[index]=0
      else:
        k=(10000.0*anf*aep)/aef
        zoltar[index]=(aef*1.0)/(aef+anf+aep+k)
    spectrum=zoltar
  elif metric=='wong-ii':
    wong=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data2):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      wong[index]=aef-aep
    spectrum=wong
  elif metric=='ochiai':
    ochiai=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data2):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      try:
        ochiai[index]=aef/np.sqrt((aef+anf)*(aef+aep))
      except: ochiai[index]=0
      # if np.isnan(ochiai[index]):
      #   print('aef:{}'.format(aef))
      #   print('anf:{}'.format(anf))
      #   print('anp:{}'.format(anp))
      #   print('aep:{}'.format(aep))
    spectrum=ochiai
  elif metric=='tarantula':
    tarantula=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data2):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      try: 
        tarantula[index]=(aef/(aef+anf))/(aef/(aef+anf)+anp/(aep+anp))
      except: tarantula[index]=0
    spectrum=tarantula
  else:
    raise Exception('The measure is not supported: {0}'.format(metric))

  if len(sp) == 2:
    channels = 1
  else:
    channels = sp[2]

  spectrum_flags=np.zeros(sp, dtype=bool)
  
  if channels > 1:
    for iindex, _ in np.ndenumerate(spectrum): # For each pixel
      tot=0
      # For each colour channel
      for j in range(0, channels):
        if not spectrum_flags[iindex[0]][iindex[1]][j]:
          tot+=spectrum[iindex[0]][iindex[1]][j] # total over channels, per pixel
      for j in range(0, channels):
        if not spectrum_flags[iindex[0]][iindex[1]][j]:
          spectrum_flags[iindex[0]][iindex[1]][j]=True # Tot has been calc for the pixel
          spectrum[iindex[0]][iindex[1]][j]=tot # Set all channels to tot

  # to smooth TODO: Figure this out
  if channels > 1:
    smooth = np.ones(spectrum.shape)
    sI = spectrum.shape[0] # pixels dim 1
    sJ = spectrum.shape[1] # pixels dim 2
    sd = (int)(sI*(10. / 224)) # Why 224?
    #print('sd: {}'.format(sd))
    for si in range(0, spectrum.shape[0]): # Pixel dim 1
        for sj in range(0, spectrum.shape[1]): # Pixel dim 2
            for sk in range(0, spectrum.shape[2]): # Colour channel
                smooth[si][sj][sk] = np.mean(spectrum[np.max([0, si-sd]):np.min([sI, si+sd]), np.max([0,sj-sd]):np.min([sJ, sj+sd]), sk])
    spectrum = smooth

  ind=np.argsort(spectrum, axis=None) # Low to high

  return ind, spectrum


def save_stats(sbfl_element, dii, len_passing, len_failing):
  if sbfl_element.immunobert:
    # This will be an array of arrays
    origin_data = sbfl_element.x[0] # Juust the input array
    #print(origin_data)
    sp=[1,origin_data.shape[0]] # It is 1D
  else:
    origin_data=sbfl_element.x
    sp=origin_data.shape
  ef=np.zeros(sp,dtype=float)
  nf=np.zeros(sp,dtype=float)
  ep=np.zeros(sp,dtype=float)
  np_=np.zeros(sp,dtype=float)

  xs=np.array(sbfl_element.xs)

  # The difference between each generated mutant and the instance to be explained
  if sbfl_element.immunobert:
    diffs = copy.deepcopy(xs)
    for i, x in enumerate(xs):
      diffs[i][0] = x[0] - origin_data
  else:
    diffs=np.abs(xs-origin_data)
  #diffs=diffs - (1+0.05 * origin_data)
  #diffs[diffs>0]=0

  for i in range(0, len(diffs)): # TODO: Not sure what happens here
    is_adv=(sbfl_element.y!=sbfl_element.ys[i])
    if sbfl_element.immunobert:
      ds_i1 = np.array(copy.deepcopy(diffs[i])[0])
      ds_i2=np.array(copy.deepcopy(diffs[i])[0])
      
    else:
      ds_i1=diffs[i].copy()
      ds_i2=diffs[i].copy()

    ds_i1[ds_i1!=0]=1
    ds_i2[ds_i2!=0]=-1
    ds_i2[ds_i2==0]=+1 # When difference is 0, it was not masked (or originally the same value = same effect as not masking)
    ds_i2[ds_i2==-1]=0

    if is_adv: # predicted to not y
      ef=ef+ds_i1 # masked
      nf=nf+ds_i2 # not masked TODO: This makes no sense, I härled to the opposite as supposed to be by text.
      #ef=ef+ds_i2
      #nf=nf+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ef[index]+=1
      #  else:
      #    nf[index]+=1
    else:
      ep=ep+ds_i1
      np_=np_+ds_i2
      #ep=ep+ds_i2
      #np_=np_+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ep[index]+=1
      #  else:
      #    np_[index]+=1

  # Save some stats to look closer at decoys.
  stats = {'len_passing': len_passing, 'len_failing': len_failing,
        'ef':ef, 'nf':nf, 'ep':ep, 'np':np_}
  #print(stats)
  np.save(dii+'/stats', stats)
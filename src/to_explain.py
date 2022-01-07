import time
import numpy as np
from spectra_gen import *
from to_rank import *
from utils import *
from datetime import datetime
from mask import *
from torch import sigmoid, tensor
import os
import platform

def to_explain(eobj):
  print ('\n[To explain: SFL (Software Fault Localization) is used]')
  print ('  ### [Measures: {0}]'.format(eobj.measures))
  model=eobj.model
  ## to create output DI
  #print ('\n[Create output folder: {0}]'.format(eobj.outputs))
  di=eobj.outputs
  sub_mat=None
  try:
    os.system('mkdir -p {0}'.format(di))
  except: pass

  if not eobj.boxes is None:
      f = open(di+"/wsol-results.txt", "a")
      f.write('input_name   x_method    intersection_with_groundtruth\n')
      f.close()

  
  if eobj.immunobert:

    # example = {'input_ids': tensor([ 2, 15, 21,  8, 21,  9, 13, 20, 15,  9, 13, 22, 11, 14,  9, 21,  3, 15,
    #           9,  8, 15, 17, 10, 19,  9, 13,  3, 14, 21, 21, 14, 16,  5,  8, 21, 14,
    #           8,  9,  8, 21, 14, 20,  3, 28, 12, 23, 14, 28, 21,  9, 13, 22, 23, 17,
    #         23, 28,  9, 17, 13,  5, 28, 26, 21, 28, 17, 15, 28, 23, 26,  5,  9, 15,
    #           5, 28, 15, 26, 28,  3]),
    #   'token_type_ids': tensor([2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
    #           2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0,
    #           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #           0, 0, 0, 0, 0, 0]),
    #   'position_ids': tensor([  0,  15,  14,  13,  12,  11,  10,   9,   8,   7,   6,   5,   4,   3,
    #             2,   1,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   0,   1,
    #             2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
    #             0,   7,   9,  24,  45,  59,  62,  63,  66,  67,  69,  70,  73,  74,
    #             76,  77,  80,  81,  84,  95,  97,  99, 114, 116, 118, 143, 147, 150,
    #           152, 156, 158, 159, 163, 167, 171,   0]),
    #   'input_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #           1, 1, 1, 1, 1, 1]),
    #   'targets': tensor([1.])}

    # eobj.inputs=[example]

    blosum_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + '..' + os.sep + 'blosum.txt'
    sub_mat = substitution_matrix(blosum_path)

  for i in range(0, len(eobj.inputs)):
    x=eobj.inputs[i]
    y = eobj.predict(x) # TODO: Change all predictions to this once settled on this solution.

    print ('\n[Input {2}: {0} / Output Label (to Explain): {1}]'.format(eobj.fnames[i], y, i))

    ite=0
    reasonable_advs=False
    while ite<eobj.testgen_iter: # Number of test generation iterations. Default 1
      print ('  #### [Start generating SFL spectra...]')
      start=time.time()
      ite+=1

      if eobj.immunobert:
        passing, failing, indexes = spectra_gen_immunobert(eobj, x, y, mask_sizes=[1,2,3], substitution_matrix=sub_mat, testgen_size=eobj.testgen_size)
      else:
        passing, failing=spectra_sym_gen(eobj, x, y, substitution_matrix=sub_mat, adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
      spectra=[]
      num_advs=len(failing)
      adv_xs=[]
      adv_ys=[]
      for e in passing:
        adv_xs.append(e)
        adv_ys.append(0)
      for e in failing:
        adv_xs.append(e)
        adv_ys.append(-1)
      tot=len(adv_xs)

      print('passing: ' + str(len(passing)))
      print('failing: ' + str(len(failing)))

      adv_part=num_advs*1./tot
      #print ('###### adv_percentage:', adv_part, num_advs, tot)
      end=time.time()
      print ('  #### [SFL spectra generation DONE: passing {0:.2f} / failing {1:.2f}, total {2}; time: {3:.0f} seconds]'.format(1-adv_part, adv_part, tot, end-start))

      if adv_part<=eobj.adv_lb:
        print ('  #### [too few failing tests: SFL explanation aborts]') 
        continue
      elif adv_part>=eobj.adv_ub:
        print ('  #### [too few many tests: SFL explanation aborts]') 
        continue
      else: 
        reasonable_advs=True
        break

    if not reasonable_advs:
      #print ('###### failed to explain')
      continue

    ## to obtain the ranking for Input i
    selement=sbfl_elementt(x, 0, adv_xs, adv_ys, model, eobj.fnames[i], immunobert=eobj.immunobert)
    dii=di+'/{0}'.format(str(datetime.now()).replace(' ', '-'))
    dii=dii.replace(':', '-')
    os.system('mkdir -p {0}'.format(dii))
    # Save the instance so we know which part is peptide, hla etc.
    if not os.path.isdir(dii):
        os.makedirs(dii)
    np.save(dii+'/instance', x)

    for measure in eobj.measures:
      print ('  #### [Measuring: {0} is used]'.format(measure))
      ranking_i, spectrum=to_rank(selement, measure)
      selement.y = y
      diii=dii+'/{0}'.format(measure)
      print ('  #### [Saving: {0}]'.format(diii))
      os.system('mkdir -p {0}'.format(diii))
      if not os.path.isdir(diii):
        os.makedirs(diii)
      np.savetxt(diii+'/ranking.txt', ranking_i, fmt='%s')
      save_spectrum(spectrum, diii+'/spectrum.txt')

      #if eobj.immunobert:
      #  immunobert_spectrum_plot(selement.x, spectrum, diii+'/'+measure+'.png')

      if not eobj.immunobert:

        # to plot the heatmap
        spectrum = np.array((spectrum/spectrum.max())*255)
        gray_img = np.array(spectrum[:,:,0],dtype='uint8')
        #print (gray_img)
        heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        if x.shape[2]==1:
            x3d = np.repeat(x[:, :, 0][:, :, np.newaxis], 3, axis=2)
        else: x3d = x
        fin = cv2.addWeighted(heatmap_img, 0.7, x3d, 0.3, 0)
        plt.rcParams["axes.grid"] = False
        plt.imshow(cv2.cvtColor(fin, cv2.COLOR_BGR2RGB))
        plt.savefig(diii+'/heatmap_{0}.png'.format(measure))

        # to plot the top ranked pixels
        if not eobj.text_only:
            ret=top_plot(selement, ranking_i, diii, measure, eobj)
            if not eobj.boxes is None:
                f = open(di+"/wsol-results.txt", "a")
                f.write('{0} {1} {2}\n'.format(eobj.fnames[i], measure, ret))
                f.close()


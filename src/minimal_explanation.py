"""
I want it to take the path to a collection of deepcover instance folders, 
go into each measure 

I need to load immunobert and so need the paths for that as well. 

I think I wasnt to do this only for the non-decoys..."""

import sys
import argparse
import os
from utils import *
from tqdm import tqdm


def main():
  parser=argparse.ArgumentParser(description='To explain neural network decisions' )
  parser.add_argument(
    '--model', dest='model', default='-1', help='the input neural network model (.h5) or (.ckpt)')
  parser.add_argument("--inputs", dest="inputs", default="-1",
                    help="the input test data directory", metavar="DIR")
  parser.add_argument("--outputs", dest="outputs", default="outs",
                    help="the output test data directory", metavar="DIR")
  parser.add_argument("--path_to_immunobert", dest='immunobert_path', help="Path to ImmunoBERT folder", metavar="DIR")
  parser.add_argument("--batch_size", dest='batch_size', help="Batch size", metavar="INT", default=1)
  parser.add_argument("--batch_number", dest='batch_number', help="Batch number", metavar="INT", default=0)
  parser.add_argument("--measures", dest="measures", default=['tarantula', 'zoltar', 'ochiai', 'wong-ii'],
                    help="the SFL measures (tarantula, zoltar, ochiai, wong-ii)", metavar="" , nargs='+')

  args=parser.parse_args()

  # Assign the arguments

  # Load the model
  sys.path.append(args.immunobert_path + os.path.sep + 'epitope')

  import pMHC
  from pMHC.logic import PresentationPredictor
  from pMHC.data.utils import convert_example_to_batch, move_dict_to_device

  pMHC.set_paths(args.immunobert_path) 
  dnn = PresentationPredictor.load_from_checkpoint(args.model,
                                                num_workers=0, shuffle_data=False, output_attentions=False)
  dnn.to("cuda")
  model = Model(dnn, convert_example_to_batch, move_dict_to_device)

  # Prep output folder
  if not os.path.isdir(args.outputs):
    os.makedirs(args.outputs)

  # Read the instance
  instance_folders = os.listdir(args.inputs)
  #instance_path = args.inputs + os.sep + instance_folders[1]

  num_included = 0
  num_excluded = 0
  avg_num_maskings = {}
  num_positive = 0
  num_negative = 0

  peptide_masks = {}#np.zeros(9)
  mhc_masks = {}#np.zeros(34)
  for measure in args.measures:
    peptide_masks[measure] = np.zeros(9)
    mhc_masks[measure] = np.zeros(34)
    avg_num_maskings[measure] = 0

  for idx, instance_folder in tqdm(enumerate(instance_folders)):
    #print(f'Instance number {idx}')
    instance_path = args.inputs + os.sep + instance_folder
    if os.path.isfile(instance_path + os.sep + 'instance.npy'):
      # What is the original prediction? 
      x = np.load(instance_path + os.sep + 'instance.npy', allow_pickle=True)
      y = model.predict(x)
      #print(f'Original predicted as: {y}')

      # Get null mask
      x2 = copy.deepcopy(x)
      x2 = total_mask(x)
      y2 = model.predict(x2)
      #print(f'Masked predicted as: {y2}')

      if y != y2:
        num_included += 1
        if y == 1:
          num_positive += 1
        else:
          num_negative += 1

        # Read the importances
        for measure in args.measures:
          #measure = 'tarantula'
          ranking = np.loadtxt(instance_path + os.sep + measure + os.sep + 'ranking.txt')
          x2 = copy.deepcopy(x)
          x2 = total_mask(x)

          # Start adding positions
          for i, position in enumerate(reversed(ranking)):
            position = int(position)
            x2[3][position] = 1
            y3 = model.predict(x2)
            if y2 != y3:
              avg_num_maskings[measure] += i + 1
              #print(f'New predict: {y3} after {i} unmaskings.')

              # Save the minimal explanation for the instance:
              instance_save_path = instance_path + os.sep + measure + os.sep + 'minimal_explanation'
              #print(f'\nSave instance explanation to: {instance_save_path}')
              np.save(instance_save_path, x2)

              # Save which ones are masked for some stats:
              peptide_positions, mhc_positions = get_positions(x2)
              peptide_masks[measure] += np.array(x2[3])[peptide_positions]
              mhc_masks[measure] += np.array(x2[3])[mhc_positions]

              break
      else: 
        num_excluded += 1

  stats = ''
  stats += f'Number of instances included: {num_included} and excluded: {num_excluded}\n'
  stats += f'Number of positive instances included: {num_positive} and negative: {num_negative}\n\n'
  for measure in args.measures:
    stats += f'Average number of unmaskings for those included ' + measure + f': {avg_num_maskings[measure]/num_included}\n'
  print(stats)
  for measure in args.measures:
    peptide_masks[measure] = [int(m) for m in peptide_masks[measure]]
    mhc_masks[measure] = [int(m) for m in mhc_masks[measure]]
    print(f'Total peptide mask {measure}: {repr(peptide_masks[measure])}')
    print(f'Total MHC mask {measure}: {repr(mhc_masks[measure])}')

  # Save the stats:
  stat_path = args.inputs + os.sep + '../minimal_explanation.txt'
  print(f'Save stats to: {stat_path}')
  with open(stat_path, 'a') as f:
    f.write(stats)
    for measure in args.measures: 
      f.write('\nPeptide and mhc masks ' + measure + ': \n')
      np.savetxt(f, peptide_masks[measure], newline=" ", fmt='%i')
      f.write('\n')
      np.savetxt(f, mhc_masks[measure], newline=" ", fmt='%i')




def get_positions(x):
    peptide_positions = (x[1] == 2) & (x[0] != 2) & (x[0] != 3)
    mhc_positions = (x[1] == 0) & (x[0] != 2) & (x[0] != 3)
    return peptide_positions, mhc_positions

class Model():
  def __init__(self, model, convert_example_to_batch, move_dict_to_device):
    self.model=model
    self.convert_example_to_batch = convert_example_to_batch
    self.move_dict_to_device = move_dict_to_device

  def predict(self, x):
    res=float(sigmoid(self.model(self.move_dict_to_device(self.convert_example_to_batch((make_instance(x))), self.model))))
    y = int(np.round(res))
    return y

# I will define masking all as masking all except 0,2,3 in the input ids. 
def total_mask(x):
    x[3] = list(map(int, (x[0] == 0) | (x[0] == 2) | (x[0] == 3)))
    #x['input_mask'] = tensor(list(map(int, (x['input_ids'] == 0) | (x['input_ids'] == 2) | (x['input_ids'] == 3))))
    return x

if __name__=="__main__":
  main()
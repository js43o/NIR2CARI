import subprocess

def main():
  a = subprocess.run(['python', 'models/pix2pixHD/test.py', '--checkpoints_dir', 'models/pix2pixHD/checkpoints', '--dataroot', 'datasets', '--results_dir', 'temp_output'])
  b = subprocess.run(['python', 'models/VToonify/style_transfer.py', '--input_path', 'temp_output', '--ckpt', 'models/VToonify/checkpoint/vtoonify_t.pt', '--style_encoder_path', 'models/VToonify/checkpoint/encoder.pt', '--faceparsing_path', 'models/VToonify/checkpoint/faceparsing.pth', '--output_path', 'output'])
  
  print(a, b)
  return

main()
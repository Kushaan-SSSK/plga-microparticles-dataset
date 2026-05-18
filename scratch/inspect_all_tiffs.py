from PIL import Image
import os
import glob

base_path = r'c:\Users\kusha\Downloads\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\PLGA_Paper\Figure'
files = glob.glob(os.path.join(base_path, '*.tif'))

for path in files:
    f = os.path.basename(path)
    try:
        img = Image.open(path)
        print(f"{f}: size={img.size}, dpi={img.info.get('dpi')}, compression={img.info.get('compression')}, mode={img.mode}")
    except Exception as e:
        print(f"{f}: error={e}")

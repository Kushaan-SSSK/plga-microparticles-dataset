from PIL import Image
import os

files = ['Fig_1.tif', 'Fig_3.tif', 'Fig_4.tif']
base_path = r'c:\Users\kusha\Downloads\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\PLGA_Paper\Figure'

for f in files:
    path = os.path.join(base_path, f)
    try:
        img = Image.open(path)
        print(f"{f}: size={img.size}, dpi={img.info.get('dpi')}, compression={img.info.get('compression')}, mode={img.mode}")
    except Exception as e:
        print(f"{f}: error={e}")

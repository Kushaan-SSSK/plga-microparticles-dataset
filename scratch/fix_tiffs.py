from PIL import Image
import os

files = ['Fig_1', 'Fig_3', 'Fig_4']
base_path = r'c:\Users\kusha\Downloads\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\A Dataset on Formulation Parameters and Characteristics of Drug-Loaded PLGA Microparticles\PLGA_Paper\Figure'

for f_name in files:
    # Source is the .tif if it exists, otherwise the .png
    tif_path = os.path.join(base_path, f_name + '.tif')
    png_path = os.path.join(base_path, f_name + '.png')
    
    if os.path.exists(tif_path):
        src_path = tif_path
    elif os.path.exists(png_path):
        src_path = png_path
    else:
        print(f"Could not find {f_name}")
        continue
        
    try:
        img = Image.open(src_path)
        # Convert to RGB to remove alpha channel (transparency)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save with LZW compression and 600 DPI
        output_path = os.path.join(base_path, f_name + '_fixed.tif')
        img.save(output_path, format='TIFF', dpi=(600, 600), compression='tiff_lzw')
        print(f"Fixed {f_name}: saved to {output_path}")
        
        # Replace original
        os.replace(output_path, tif_path)
        print(f"Successfully replaced {tif_path}")
        
    except Exception as e:
        print(f"Error fixing {f_name}: {e}")

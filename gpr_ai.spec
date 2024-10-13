# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Análise do script principal e dependências
a = Analysis(['gpr_ai.py'],
             pathex=[],
             binaries=[],
             datas=[('cnn_model.py', '.'), ('add_images_to_dataset.py', '.'), ('organize_images.py', '.')],
             hiddenimports=['tensorflow', 'cv2', 'sklearn', 'PIL', 'matplotlib'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# Criação do arquivo PYZ
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Criação do executável
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='GPR_AI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None)

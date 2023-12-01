from setuptools import setup


setup(
    name='ttsxai',
    description='',
    # packages=['sxai', 'torchattacks', 'tacotron2', 'waveglow'],
    # packages=['sxai', 'mellotron', 'tacotron2', 'waveglow'],
    # package_dir={'':'src'},
    packages=[
        'ttsxai', 
        'tacotron2', 
        'fastspeech2', 
        'mellotron', 
        'flowtron',
        'waveglow',
        'neurox'
    ],
    package_dir={'':'src'},
    install_requires=[
        'matplotlib==3.5.1',
        'librosa==0.9.2',
        'praat-parselmouth~=0.4.2',
        'torch==1.13.1', # 'torch==2.0.0',
        'scipy==1.7.3', # 'scipy==1.9.3',
        'pyloudnorm==0.1.0',
        'torchaudio==0.13.1', # 'torchaudio==2.0.0',
        'wandb==0.13.5',
        'dragonmapper==0.2.6',
        'phonemizer==3.2.1',
        'pypinyin==0.47.1',
        'pyloudnorm==0.1.0',
        'alias_free_torch==0.0.6',
        'speechbrain==0.5.13',
        'torch_complex==0.4.3',
        'g2p-en',
        # === manifold ===
        'einops==0.6.0',
        # === fastspeech 2 ===
        'tgt==1.4.4',
        'pyworld==0.2.11.post0',
        # === tacotron2 === 
        'tensorflow==1.15.2',
        # === tensorboard error ===
        'protobuf==3.19.0',
        # === neurox related ===
        'imblearn'
    ]
)
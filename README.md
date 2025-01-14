# SAE 5: Génération d'images avec Stable Diffusion

## Description
Cette SAE à pour but de mettre en place un générateur d'images pour une application de réseau social. L'application permet de créer des avatars ou des illustrations à l'aide du modèle **Stable Diffusion**. Plusieurs techniques avancées de génération d'images sont implémentées, permettant une grande flexibilité dans la création des images.

## Fonctionnalités
L'application propose plusieurs techniques de génération d'images :

1. **Text2Image** : Génération d'images à partir de texte.
   - L'utilisateur peut choisir parmi plusieurs styles (noir et blanc, portrait, etc.).
   - Utilisation de l'ingénierie de prompt pour améliorer la qualité des images.

2. **Image2Image** : Génération d'une nouvelle image basée sur une image existante et un prompt.
   - L'utilisateur peut choisir différentes stratégies pour transformer l'image.

3. **Inpainting** : Modification locale d'une image existante (suppression d'objets, modification de parties).
   - L'utilisateur peut fournir un masque ou le générer automatiquement à partir du prompt.

4. **Dreambooth et LoRA** : Fine-tuning du modèle pour ajouter des objets personnalisés dans les images.
   - L'utilisateur doit fournir quelques images personnelles pour l'entraînement.

5. **Dreambooth + Inpainting** : Combinaison de fine-tuning et d'Inpainting pour personnaliser des parties spécifiques d'une image existante.

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/DamienAllaert/SAE5.git
   ```
   
2. Installer les prérequis :

   ```bash
   !pip install torch 
   ```
   ```bash
   !pip install diffusers==0.30.0 transformers==4.45.1 scipy ftfy accelerate
   ```
   ```bash
   !pip install -qq bitsandbytes
   ```
   ```bash
   !pip install -Uq controlnet-aux
    ```
    ```bash
   !pip install mmengine
    ```
    ```bash
   !pip install git-lfs
    ```
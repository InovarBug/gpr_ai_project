import os
import shutil
from sklearn.model_selection import train_test_split


def organize_dataset(source_dir, dest_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    # Lista todas as classes no diretório de origem
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in classes:
        # Cria diretórios para cada classe em train, validation e test
        os.makedirs(os.path.join(dest_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'validation', class_name), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'test', class_name), exist_ok=True)

        # Lista todas as imagens na classe
        class_dir = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if len(images) == 0:
            print(f"Aviso: Nenhuma imagem encontrada na classe {class_name}")
            continue

        # Divide as imagens em conjuntos de treino, validação e teste
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size / (train_size + val_size), random_state=42)

        # Função para copiar imagens
        def copy_images(image_list, subset):
            for img in image_list:
                src = os.path.join(class_dir, img)
                dst = os.path.join(dest_dir, subset, class_name, img)
                shutil.copy2(src, dst)

        # Copia as imagens para os respectivos diretórios
        copy_images(train, 'train')
        copy_images(val, 'validation')
        copy_images(test, 'test')

        print(f"Classe {class_name}: {len(train)} treino, {len(val)} validação, {len(test)} teste")

    print("Organização do dataset concluída.")
    return {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size
}

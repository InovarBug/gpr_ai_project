import os
import shutil
from tkinter import filedialog, simpledialog, Tk, messagebox

def select_images():
    # Abre um diálogo para selecionar imagens
    root = Tk()
    root.withdraw()  # Oculta a janela principal do Tkinter
    image_files = filedialog.askopenfilenames(
        title="Selecione as Imagens",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    return image_files

def select_dataset_dir():
    # Abre um diálogo para selecionar o diretório do dataset
    root = Tk()
    root.withdraw()
    dataset_dir = filedialog.askdirectory(title="Selecione o Diretório do Dataset")
    return dataset_dir

def organize_images_into_dataset():
    # Organiza as imagens selecionadas no dataset
    image_files = select_images()
    if not image_files:
        messagebox.showerror("Erro", "Nenhuma imagem selecionada.")
        return

    dataset_dir = select_dataset_dir()
    if not dataset_dir:
        messagebox.showerror("Erro", "Nenhum diretório de dataset selecionado.")
        return

    class_name = simpledialog.askstring("Classe", "Digite o nome da classe para essas imagens:")
    if not class_name:
        messagebox.showerror("Erro", "Nome da classe não fornecido.")
        return

    dataset_type = simpledialog.askstring("Conjunto de Dados", "Digite o conjunto para essas imagens (train, validation, ou test):")
    if dataset_type not in ['train', 'validation', 'test']:
        messagebox.showerror("Erro", "Conjunto inválido. Use 'train', 'validation', ou 'test'.")
        return

    class_dir = os.path.join(dataset_dir, dataset_type, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for image_path in image_files:
        filename = os.path.basename(image_path)
        destination = os.path.join(class_dir, filename)
        shutil.copy2(image_path, destination)

    messagebox.showinfo("Sucesso", f"{len(image_files)} imagens adicionadas à classe '{class_name}' no conjunto '{dataset_type}'.")

if __name__ == "__main__":
    organize_images_into_dataset()

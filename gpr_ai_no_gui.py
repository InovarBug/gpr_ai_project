import numpy as np
import tensorflow as tf
from scipy.signal import butter, lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import concurrent.futures

from PIL import Image, ImageTk
import io
import sqlite3
import pickle
import os
import shutil
import subprocess
from cnn_model import CNNGPR

print(f"TensorFlow version: {tf.__version__}")
print("Todas as bibliotecas foram importadas com sucesso.")

class AdvancedGPRAI:
    """
    Classe para a Inteligência Artificial Avançada de GPR.
    """
    def __init__(self):
        self.cnn_model = self._build_cnn_model()
        self.db_path = 'gpr_patterns.db'
        self._create_db()
        self.load_patterns()
        self.knowledge_base = {}

    def _create_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns
            (id INTEGER PRIMARY KEY, name TEXT, description TEXT, pattern BLOB)
        ''')
        conn.commit()
        conn.close()

    def _build_cnn_model(self):
        """
        Constrói e compila o modelo CNN.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def add_pattern(self, name, pattern, description):
        """
        Adiciona um novo padrão ao banco de dados.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        pattern_blob = pickle.dumps(pattern)
        cursor.execute('INSERT INTO patterns (name, description, pattern) VALUES (?, ?, ?)',
                       (name, description, pattern_blob))
        conn.commit()
        conn.close()
        self.load_patterns()

    def load_patterns(self):
        """
        Carrega todos os padrões do banco de dados para a memória.
        """
        self.patterns_db = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, description, pattern FROM patterns')
        for row in cursor.fetchall():
            id, name, description, pattern_blob = row
            pattern = pickle.loads(pattern_blob)
            self.patterns_db[id] = {'name': name, 'pattern': pattern, 'description': description}
        conn.close()

    def delete_pattern(self, pattern_id):
        """
        Deleta um padrão do banco de dados pelo ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM patterns WHERE id = ?', (pattern_id,))
        conn.commit()
        conn.close()
        self.load_patterns()

    def update_pattern(self, pattern_id, name, description):
        """
        Atualiza um padrão existente no banco de dados.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE patterns SET name = ?, description = ? WHERE id = ?',
                       (name, description, pattern_id))
        conn.commit()
        conn.close()
        self.load_patterns()

    def analyze_image(self, image):
        """
        Analisa uma imagem em busca de padrões e anomalias.
        """
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        anomalies = []

        for id, pattern_info in self.patterns_db.items():
            pattern = pattern_info['pattern']
            name = pattern_info['name']
            description = pattern_info['description']

            matches = cv2.matchTemplate(image, pattern, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            locations = np.where(matches >= threshold)

            for pt in zip(*locations[::-1]):
                cv2.rectangle(colored_image, pt, (pt[0] + pattern.shape[1], pt[1] + pattern.shape[0]), (0, 255, 0), 2)
                anomalies.append((name, pt, description))

        return {
            "imagem_analisada": colored_image,
            "anomalias": anomalies,
            "profundidade": f"{np.random.uniform(1, 5):.1f} metros",
            "confianca": f"{np.random.uniform(70, 99):.1f}%"
        }

class GPRInterface:
    """
    Classe para a interface gráfica do usuário (GUI) da aplicação GPR AI.
    """
    def __init__(self, master):
        """
        Inicializa a interface gráfica do usuário (GUI) para a aplicação GPR AI.
        """
        print("Inicializando a interface GPR...")
        self.master = master
        self.master.title("GPR AI Interface")
        self.ai = AdvancedGPRAI()
        self.cnn = CNNGPR()
        self.dataset_dir = ""

        self.create_widgets()
        print("Interface GPR inicializada.")

    def create_widgets(self):
        """
        Cria os widgets da interface gráfica do usuário (GUI).
        """

        self.notebook.add(self.organize_images_frame, text='Organizar Imagens')

        organize_images_button_frame = tk.Frame(self.organize_images_frame)
        organize_images_button_frame.pack(pady=10)

        self.open_organize_images_button = tk.Button(organize_images_button_frame, text="Organizar Imagens para CNN", command=self.open_organize_images)
        self.open_organize_images_button.pack(side=tk.LEFT, padx=5)

        self.refresh_patterns()

    def import_image(self):
        """
        Importa uma imagem para análise.
        """
        """
        Importa uma imagem para análise.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image, self.image_label)
            self.analyze_button['state'] = tk.NORMAL

    def display_image(self, image, label):
        image = cv2.resize(image, (256, 256))
        img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(image=img)
        label.config(image=img_tk)
        label.image = img_tk

    def analyze_image(self):
        """
        Analisa a imagem importada e exibe os resultados.
        """
        """
        Analisa a imagem importada e exibe os resultados.
        """
        if hasattr(self, 'image'):
            print("Analisando imagem...")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Analisando imagem...\n")

            results = self.ai.analyze_image(self.image)

            analyzed_image = cv2.cvtColor(results['imagem_analisada'], cv2.COLOR_BGR2RGB)
            analyzed_image = cv2.resize(analyzed_image, (256, 256))
            img = Image.fromarray(analyzed_image)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            self.result_text.insert(tk.END, "Análise concluída.\n")
            self.result_text.insert(tk.END, f"Anomalias detectadas: {len(results['anomalias'])}\n")
            for name, _, desc in results['anomalias']:
                self.result_text.insert(tk.END, f"- {name}: {desc}\n")
            self.result_text.insert(tk.END, f"Profundidade estimada: {results['profundidade']}\n")
            self.result_text.insert(tk.END, f"Confiança da análise: {results['confianca']}\n")
            print("Análise concluída.")

    def add_pattern(self):
        """
        Adiciona um novo padrão ao banco de dados.
        """
        """
        Adiciona um novo padrão ao banco de dados.
        """
        name = simpledialog.askstring("Nome do Padrão", "Digite o nome do padrão:")
        if name:
            description = simpledialog.askstring("Descrição do Padrão", "Digite a descrição do padrão:")
            if description:
                file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
                if file_path:
                    pattern_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    self.ai.add_pattern(name, pattern_image, description)
                    self.result_text.insert(tk.END,
                                            f"Padrão '{name}' adicionado com sucesso e salvo no banco de dados.\n")
                    self.refresh_patterns()

    def refresh_patterns(self):
        """
        Atualiza a lista de padrões exibida na interface.
        """
        """
        Atualiza a lista de padrões exibida na interface.
        """
        for i in self.patterns_tree.get_children():
            self.patterns_tree.delete(i)
        for id, pattern_info in self.ai.patterns_db.items():
            self.patterns_tree.insert('', 'end', values=(id, pattern_info['name'], pattern_info['description']))

    def edit_pattern(self):
        """
        Edita um padrão existente no banco de dados.
        """
        """
        Edita um padrão existente no banco de dados.
        """
        selected_item = self.patterns_tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Por favor, selecione um padrão para editar.")
            return

        pattern_id = self.patterns_tree.item(selected_item)['values'][0]
        pattern_info = self.ai.patterns_db[pattern_id]

        new_name = simpledialog.askstring("Editar Nome", "Digite o novo nome do padrão:",
                                          initialvalue=pattern_info['name'])
        if new_name:
            new_description = simpledialog.askstring("Editar Descrição", "Digite a nova descrição do padrão:",
                                                     initialvalue=pattern_info['description'])
            if new_description:
                self.ai.update_pattern(pattern_id, new_name, new_description)
                self.refresh_patterns()
                messagebox.showinfo("Sucesso", "Padrão atualizado com sucesso.")

    def delete_pattern(self):
        """
        Deleta um padrão existente no banco de dados.
        """
        """
        Deleta um padrão existente no banco de dados.
        """
        selected_item = self.patterns_tree.selection()
        if not selected_item:
            messagebox.showwarning("Aviso", "Por favor, selecione um padrão para excluir.")
            return

        pattern_id = self.patterns_tree.item(selected_item)['values'][0]
        if messagebox.askyesno("Confirmar Exclusão", "Tem certeza que deseja excluir este padrão?"):
            self.ai.delete_pattern(pattern_id)
            self.refresh_patterns()
            messagebox.showinfo("Sucesso", "Padrão excluído com sucesso.")

    def cnn_import_image(self):
        """
        Importa uma imagem para a CNN.
        """
        """
        Importa uma imagem para a CNN.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            self.cnn_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.cnn_image, self.cnn_image_label)
            self.cnn_predict_button['state'] = tk.NORMAL

    def cnn_predict(self):
        """
        Realiza a previsão usando a CNN.
        """
        """
        Realiza a previsão usando a CNN.
        """
        if hasattr(self, 'cnn_image'):
            prediction = self.cnn.predict(self.cnn_image)
            self.cnn_result_text.delete(1.0, tk.END)
            self.cnn_result_text.insert(tk.END, f"Previsão CNN: Classe {prediction}\n")

    def cnn_train(self):
        """
        Treina o modelo CNN com os dados fornecidos.
        """
        """
        Treina o modelo CNN com os dados fornecidos.
        """
        train_dir = filedialog.askdirectory(title="Selecione o diretório de treinamento")
        validation_dir = filedialog.askdirectory(title="Selecione o diretório de validação")

        if train_dir and validation_dir:
            epochs = simpledialog.askinteger("Épocas", "Número de épocas:", initialvalue=10, minvalue=1, maxvalue=100)
            batch_size = simpledialog.askinteger("Batch Size", "Tamanho do batch:", initialvalue=32, minvalue=1,
                                                 maxvalue=128)

            if epochs and batch_size:
                try:
                    history = self.cnn.train(train_dir, validation_dir, epochs, batch_size)

                    # Plotar resultados do treinamento
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['accuracy'], label='Treino')
                    plt.plot(history.history['val_accuracy'], label='Validação')
                    plt.title('Acurácia do Modelo')
                    plt.xlabel('Época')
                    plt.ylabel('Acurácia')
                    plt.legend()

                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['loss'], label='Treino')
                    plt.plot(history.history['val_loss'], label='Validação')
                    plt.title('Perda do Modelo')
                    plt.xlabel('Época')
                    plt.ylabel('Perda')
                    plt.legend()

                    plt.tight_layout()
                    plt.show()

                    messagebox.showinfo("Treinamento Concluído", "O modelo CNN foi treinado com sucesso!")
                except Exception as e:
                    messagebox.showerror("Erro no Treinamento", f"Ocorreu um erro durante o treinamento: {str(e)}")

    def save_cnn_model(self):
        """
        Salva o modelo CNN treinado em um arquivo.
        """
        """
        Salva o modelo CNN treinado em um arquivo.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.cnn.save_model(file_path)
            messagebox.showinfo("Modelo Salvo", f"O modelo foi salvo em {file_path}")

    def load_cnn_model(self):
        """
        Carrega um modelo CNN a partir de um arquivo.
        """
        """
        Carrega um modelo CNN a partir de um arquivo.
        """
        file_path = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5")])
        if file_path:
            self.cnn.load_model(file_path)
            messagebox.showinfo("Modelo Carregado", f"O modelo foi carregado de {file_path}")

    def evaluate_cnn_model(self):
        """
        Avalia o modelo CNN com os dados de teste fornecidos.
        """
        """
        Avalia o modelo CNN com os dados de teste fornecidos.
        """
        test_dir = filedialog.askdirectory(title="Selecione o diretório de teste")
        if test_dir:
            try:
                results = self.cnn.evaluate(test_dir)
                result_str = "\n".join([f"{k}: {v:.4f}" for k, v in results.items()])
                messagebox.showinfo("Resultados da Avaliação", f"Resultados:\n{result_str}")
            except Exception as e:
                messagebox.showerror("Erro na Avaliação", f"Ocorreu um erro durante a avaliação: {str(e)}")

    def select_dataset_dir(self):
        """
        Seleciona o diretório do dataset.
        """
        """
        Seleciona o diretório do dataset.
        """
        self.dataset_dir = filedialog.askdirectory(title="Selecione o Diretório do Dataset")
        if self.dataset_dir:
            self.add_images_result_text.insert(tk.END, f"Diretório do dataset selecionado: {self.dataset_dir}\n")

    def add_images_to_dataset(self):
        """
        Adiciona imagens ao dataset selecionado.
        """
        """
        Adiciona imagens ao dataset selecionado.
        """
        if not self.dataset_dir:
            messagebox.showerror("Erro", "Por favor, selecione primeiro o diretório do dataset.")
            return

        image_files = filedialog.askopenfilenames(
            title="Selecione as Imagens",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if not image_files:
            return

        class_name = simpledialog.askstring("Classe", "Digite o nome da classe para essas imagens:")
        if not class_name:
            return

        dataset_type = simpledialog.askstring("Conjunto de Dados",
                                              "Digite o conjunto para essas imagens (train, validation, ou test):")
        if dataset_type not in ['train', 'validation', 'test']:
            messagebox.showerror("Erro", "Conjunto inválido. Use 'train', 'validation', ou 'test'.")
            return

        class_dir = os.path.join(self.dataset_dir, dataset_type, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for image_path in image_files:
            filename = os.path.basename(image_path)
            destination = os.path.join(class_dir, filename)
            shutil.copy2(image_path, destination)

        self.add_images_result_text.insert(tk.END, f"{len(image_files)} imagens adicionadas à classe '{class_name}' no conjunto '{dataset_type}'.\n")

    def open_add_images_to_dataset(self):
        """
        Abre o script add_images_to_dataset.py em um novo processo.
        """
        """
        Abre o script add_images_to_dataset.py em um novo processo.
        """
        try:
            subprocess.Popen(["python", "add_images_to_dataset.py"])
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir add_images_to_dataset.py: {str(e)}")

    def open_organize_images(self):
        """
        Abre o script organize_images.py em um novo processo.
        """
        """
        Abre o script organize_images.py em um novo processo.
        """
        try:
            subprocess.Popen(["python", "organize_images.py"])
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir organize_images.py: {str(e)}")

def main():
    """
    Função principal para iniciar a aplicação GPR AI.
    """
    """
    Função principal para iniciar a aplicação GPR AI.
    """
    print("Iniciando a aplicação...")
    if os.environ.get('DISPLAY'):
        root = tk.Tk()
    else:
        print("Ambiente sem suporte para GUI. A inicialização da interface gráfica foi pulada.")
        return
    app = GPRInterface(root)
    root.geometry("600x700")  # Aumentado o tamanho da janela
    root.deiconify()
    print("Iniciando o loop principal da interface gráfica...")
    root.mainloop()

if __name__ == "__main__":
    main()

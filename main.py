from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.filechooser import FileChooserListView, FileChooserIconView
from kivymd.app import MDApp
from kivy.uix.popup import Popup
from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDIconButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.card import MDCard
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivy.uix.button import Button
import threading
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.toolbar import MDTopAppBar
from kivy.uix.video import Video
from kivy.uix.videoplayer import VideoPlayer
from kivymd.uix.slider import MDSlider
from kivy.uix.image import Image
from kivymd.uix.selectioncontrol import MDCheckbox, MDSwitch
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.dialog import MDDialog
from kivymd.uix.progressbar import MDProgressBar
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ListProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color, Line
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
import os
from stitcher import get_frames, cascade_frame_search, getValores, unir_cascade_frames_manteniendo_superior
from moviepy import VideoFileClip
import tempfile
from kivymd.toast import toast
import traceback
import cv2
import json
import shutil  # Librería para copiar archivos
from pathlib import Path  # Manejo de rutas de forma más eficiente
# Configurar para móvil
Window.size = (360, 640)

class VideoSelectionScreen(Screen):
    pass

class EditScreen(Screen):
    pass

class ParametersScreen(Screen):
    pass

class ResultScreen(Screen):
    pass

class SettingScreen(Screen):
    pass

class VideoProcessingApp(MDApp):
    selected_video = StringProperty(None)
    video_duration = NumericProperty(0)
    is_playing = BooleanProperty(True)
    progress_value = NumericProperty(0)
    cut_start_time = NumericProperty(0)
    cut_end_time = NumericProperty(0)
    processed_image_path = StringProperty("")
    processed_frames = ListProperty([])
    cascade_frames = ListProperty([])
    metodo_fusion = StringProperty("")
    direccion = StringProperty("")
    
    cap = None
    total_frames = NumericProperty(0)
    fps = NumericProperty(0)
    start_frame = NumericProperty(0)
    end_frame = NumericProperty(0)
    image_widget = Image()
    crop_mode = BooleanProperty(False)
    crop_coords = None
    crop_mode_button_color = ListProperty([0, 0.5, 1, 1])

            
    
    def build(self):
        self.settings = self.load_settings()
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.accent_palette = "Amber"
        # Configurar el tema según los ajustes guardados
        self.theme_cls.theme_style = self.settings.get("theme", "Light")
        return Builder.load_file("screen.kv")

    def toggle_theme(self, is_active):
        self.theme_cls.theme_style = "Dark" if is_active else "Light"
        self.settings["theme"] = self.theme_cls.theme_style
        self.save_settings()

    def load_settings(self):
        """
        Cargar configuraciones desde un archivo JSON.
        Si el archivo no existe, se crea con propiedades predeterminadas.
        """
        default_settings = {
        "theme": "",
        "language": "es",
        "video_path": "",
        "image_path": ""
        }
        try:
            # Intentar cargar el archivo settings.json
            with open("settings.json", "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si no existe o está corrupto, crear el archivo con valores por defecto
            with open("settings.json", "w") as file:
                json.dump(default_settings, file, indent=4)
            return default_settings

    def save_settings(self):
        """
        Guardar configuraciones en un archivo JSON.
        """
        with open("settings.json", "w") as file:
            json.dump(self.settings, file)
    
    def select_video_folder(self):
        """Abre un diálogo para seleccionar la carpeta de videos"""
        # Crear el FileChooser
        filechooser = FileChooserIconView(
            path=os.path.expanduser("/"),
            dirselect=True
        )
    
        # Crear botón Aceptar
        accept_button = Button(
            text="Aceptar",
            size_hint=(0.3, 0.1),
            background_color=(0.1, 0.5, 0.8, 1)
        )
        accept_button.bind(on_release=lambda x: self.on_folder_selected(filechooser.path, popup))
    
        # Crear el layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(filechooser)
        layout.add_widget(accept_button)
    
        # Crear y mostrar el Popup
        popup = Popup(
            title="Selecciona la carpeta de videos",
            content=layout,
            size_hint=(0.9, 0.9)
        )
        popup.open()

    def on_folder_selected(self, folder_path, popup):
        """Maneja la selección de la carpeta"""
        if os.path.isdir(folder_path):
            self.set_video_folder(folder_path)
        popup.dismiss()

    def set_video_folder(self, folder_path):
        """Establece la carpeta de videos seleccionada"""
        self.settings["video_path"] = folder_path
        self.save_settings()
    
        # Actualizar la pantalla de configuración
        settings_screen = self.root.get_screen("settings")
        settings_screen.ids.video_path_label.text = folder_path
    
        # Actualizar la pantalla de selección de videos
        video_selection_screen = self.root.get_screen("video_selection")
        video_selection_screen.ids.filechooser.path = folder_path
    
    def on_directory_selected(self, filechooser, popup):
        selected_path = filechooser.path  # Carpeta seleccionada
        popup.dismiss()
    
    def set_image_folder(self, folder_path):
        """Establece la carpeta de imágenes seleccionada"""
        self.settings["image_path"] = folder_path
        self.save_settings()

        # Actualizar la pantalla de configuración
        settings_screen = self.root.get_screen("settings")
        settings_screen.ids.image_path_label.text = folder_path
    
    def select_image_folder(self):
        """Abre el selector de carpetas para elegir la carpeta de imágenes"""
        filechooser = FileChooserIconView(dirselect=True)

        accept_button = Button(text="Aceptar", size_hint=(0.3, 0.1))
        accept_button.bind(on_release=lambda x: self.on_image_folder_selected(filechooser, popup))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(filechooser)
        layout.add_widget(accept_button)

        popup = Popup(title="Selecciona una carpeta de imágenes",
                  content=layout,
                  size_hint=(0.9, 0.9))
        popup.open()
    
    def on_image_folder_selected(self, filechooser, popup):
        """Maneja la selección de la carpeta de imágenes"""
        selection = filechooser.selection
        if selection:
            folder_path = selection[0]
            self.set_image_folder(folder_path)
        popup.dismiss()
        
    
    def on_start(self):
        self.create_dropdown_menus()
    
    def create_dropdown_menus(self):
        # Filtro menu
        filtro_items = [
            {"text": f"{i}", "viewclass": "OneLineListItem", 
             "on_release": lambda x=i: self.set_filtro(x)} 
            for i in ['canny', 'sobel', 'laplacian', 'scharr']
        ]
        self.filtro_menu = MDDropdownMenu(
            caller=self.root.get_screen('parameters').ids.filtro_button,
            items=filtro_items,
            width_mult=4,
        )
        
        # Método comparación menu
        metodo_comparacion_items = [
            {"text": f"{i}", "viewclass": "OneLineListItem", 
             "on_release": lambda x=i: self.set_metodo_comparacion(x)} 
            for i in ['pixel', 'ssim', 'histograma', 'orb', 'combinado']
        ]
        self.metodo_comparacion_menu = MDDropdownMenu(
            caller=self.root.get_screen('parameters').ids.metodo_comparacion_button,
            items=metodo_comparacion_items,
            width_mult=4,
        )
        
        # Método fusión menu
        metodo_fusion_items = [
            {"text": f"{i}", "viewclass": "OneLineListItem", 
             "on_release": lambda x=i: self.set_metodo_fusion(x)} 
            for i in ['simple', 'superposicion', 'gradiente', 'costura']
        ]
        self.metodo_fusion_menu = MDDropdownMenu(
            caller=self.root.get_screen('parameters').ids.metodo_fusion_button,
            items=metodo_fusion_items,
            width_mult=4,
        )
    
    def show_filtro_menu(self):
        self.filtro_menu.open()
    
    def set_filtro(self, filtro):
        self.root.get_screen('parameters').ids.filtro_button.text = filtro
        self.filtro_menu.dismiss()
    
    def show_metodo_comparacion_menu(self):
        self.metodo_comparacion_menu.open()
    
    def set_metodo_comparacion(self, metodo):
        self.root.get_screen('parameters').ids.metodo_comparacion_button.text = metodo
        self.metodo_comparacion_menu.dismiss()
    
    def show_metodo_fusion_menu(self):
        self.metodo_fusion_menu.open()
    
    def set_metodo_fusion(self, metodo):
        self.root.get_screen('parameters').ids.metodo_fusion_button.text = metodo
        self.metodo_fusion_menu.dismiss()
    
    def select_video(self, selection):
        if selection:
            self.selected_video = selection[0]
            self.root.get_screen('video_selection').ids.selected_video_label.text = os.path.basename(self.selected_video)
            self.root.get_screen('video_selection').ids.next_button.disabled = False
    
    def go_to_edit(self):
        if self.selected_video:
            self.root.current = "edit"
            video_screen = self.root.get_screen('edit')
            
            # Obtener la duración del video
            try:
                clip = VideoFileClip(self.selected_video)
                self.video_duration = clip.duration
                clip.close()
        
                # Configurar sliders
                video_screen.ids.start_slider.value = 0
                video_screen.ids.end_slider.value = 100
                self.cut_start_time = 0
                self.cut_end_time = self.video_duration
                
                self.load_video_metadata()
                self.show_frame(0)
        
            except Exception as e:
                self.show_error_dialog(f"Error al cargar el video: {str(e)}")
                print(f"Error detallado: {str(e)}")
                traceback.print_exc()
        else:
            self.show_error_dialog("Por favor, seleccione un video antes de continuar.")
            
    def load_video_metadata(self):
        """Carga los metadatos del video seleccionado"""
        if self.cap:
            self.cap.release()
    
        self.cap = cv2.VideoCapture(self.selected_video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
        # Configurar sliders
        self.root.get_screen('edit').ids.start_slider.max = self.total_frames - 1
        self.root.get_screen('edit').ids.end_slider.max = self.total_frames - 1
        self.root.get_screen('edit').ids.end_slider.value = self.total_frames - 1
    
        # Mostrar primer frame
        self.show_frame(0)        
            
    def show_frame(self, frame_number):
        """Muestra un frame específico del video"""
        if not self.cap:
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
    
        if ret:
            self.display_frame(frame)
            self.update_time_labels(frame_number)

    def update_crop_rectangle(self):
        """Actualiza el rectángulo de recorte cuando cambia el tamaño/posición de la imagen."""
        if self.crop_mode and self.image_widget.texture:
            if self.crop_coords is None:
                self.initialize_crop_rectangle()
            else:
                self.draw_crop_rectangle()

    def display_frame(self, frame):
        """Muestra el frame en la interfaz"""
        flipped_frame = cv2.flip(frame, 0)
        frame_rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        buf = frame_rgb.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
    
        video_box = self.root.get_screen('edit').ids.video_box
        video_box.clear_widgets()
        self.image_widget.texture = texture
        self.image_widget.allow_stretch = True
        self.image_widget.keep_ratio = True
        video_box.add_widget(self.image_widget)
    
        if self.crop_mode:
            Clock.schedule_once(lambda dt: self.update_crop_rectangle(), 0.1)       
            
    def update_start_frame(self, value):
        """Actualiza el frame de inicio"""
        frame_number = int(value)
        if frame_number > self.end_frame:
            frame_number = self.end_frame
            self.root.get_screen('edit').ids.start_slider.value = frame_number
    
        self.start_frame = frame_number
        self.show_frame(frame_number)

    def update_end_frame(self, value):
        """Actualiza el frame final"""
        frame_number = int(value)
        if frame_number < self.start_frame:
            frame_number = self.start_frame
            self.root.get_screen('edit').ids.end_slider.value = frame_number
    
        self.end_frame = frame_number
        self.show_frame(frame_number)

    def update_time_labels(self, frame_number):
        """Actualiza las etiquetas de tiempo"""
        time_sec = frame_number / self.fps
        mins = int(time_sec // 60)
        secs = int(time_sec % 60)
    
        if frame_number == self.start_frame:
            self.root.get_screen('edit').ids.start_time_label.text = f"{mins:02d}:{secs:02d}"
        elif frame_number == self.end_frame:
            self.root.get_screen('edit').ids.end_time_label.text = f"{mins:02d}:{secs:02d}"  
            
    def draw_crop_rectangle(self):
        """Dibuja el rectángulo de recorte y sus manijas."""
        if not self.crop_coords or not self.image_widget.texture:
            return
            
        # Limpiar canvas anterior
        edit_screen = self.root.get_screen('edit')
        video_box = edit_screen.ids.video_box
        video_box.canvas.after.clear()
        #self.root.ids.video_box.canvas.after.clear()
        
        x1, y1, x2, y2 = self.crop_coords
        img_x, img_y, img_w, img_h = self.get_image_display_region()
        
        # Limitar las coordenadas al área de la imagen
        x1 = max(img_x, min(x1, img_x + img_w))
        y1 = max(img_y, min(y1, img_y + img_h))
        x2 = max(img_x, min(x2, img_x + img_w))
        y2 = max(img_y, min(y2, img_y + img_h))
        
        self.crop_coords = (x1, y1, x2, y2)
        
        with video_box.canvas.after:
            # Dibujar rectángulo de recorte
            Color(1, 0, 0, 0.5)
            Line(rectangle=(x1, y1, x2 - x1, y2 - y1), width=2)
            
            # Dibujar manijas en las esquinas
            Color(1, 1, 0, 1)
            handle_size = 10
            self.crop_handles = [
                Rectangle(pos=(x1 - handle_size/2, y1 - handle_size/2), size=(handle_size, handle_size)),
                Rectangle(pos=(x2 - handle_size/2, y1 - handle_size/2), size=(handle_size, handle_size)),
                Rectangle(pos=(x1 - handle_size/2, y2 - handle_size/2), size=(handle_size, handle_size)),
                Rectangle(pos=(x2 - handle_size/2, y2 - handle_size/2), size=(handle_size, handle_size))
            ]
    
    def on_touch_down(self, touch):
        """Maneja el evento de toque inicial en el modo recorte."""
        if not self.crop_mode or not self.crop_coords:
            return

        x, y = touch.pos
        x1, y1, x2, y2 = self.crop_coords
        handle_size = 10

        # Verificar si se está tocando alguna de las manijas
        if abs(x - x1) < handle_size and abs(y - y1) < handle_size:
            self.dragging_corner = 'top_left'
        elif abs(x - x2) < handle_size and abs(y - y1) < handle_size:
            self.dragging_corner = 'top_right'
        elif abs(x - x1) < handle_size and abs(y - y2) < handle_size:
            self.dragging_corner = 'bottom_left'
        elif abs(x - x2) < handle_size and abs(y - y2) < handle_size:
            self.dragging_corner = 'bottom_right'
        # Verificar si está dentro del rectángulo para moverlo completo
        elif x1 <= x <= x2 and y1 <= y <= y2:
            self.dragging_rect = True
            self.drag_start_pos = (x, y)
            self.drag_start_coords = self.crop_coords
        else:
            self.dragging_corner = None
            self.dragging_rect = False

    def on_touch_move(self, touch):
        """Maneja el movimiento del rectángulo de recorte o sus esquinas."""
        if not self.crop_mode or (not self.dragging_corner and not self.dragging_rect) or not self.crop_coords:
            return

        x, y = touch.pos
        img_x, img_y, img_w, img_h = self.get_image_display_region()
        
        # Limitar coordenadas al área de la imagen
        x = max(img_x, min(x, img_x + img_w))
        y = max(img_y, min(y, img_y + img_h))

        if self.dragging_corner:
            x1, y1, x2, y2 = self.crop_coords

            if self.dragging_corner == 'top_left':
                x1 = max(img_x, min(x, x2 - 10))
                y1 = max(img_y, min(y, y2 - 10))
            elif self.dragging_corner == 'top_right':
                x2 = min(img_x + img_w, max(x, x1 + 10))
                y1 = max(img_y, min(y, y2 - 10))
            elif self.dragging_corner == 'bottom_left':
                x1 = max(img_x, min(x, x2 - 10))
                y2 = min(img_y + img_h, max(y, y1 + 10))
            elif self.dragging_corner == 'bottom_right':
                x2 = min(img_x + img_w, max(x, x1 + 10))
                y2 = min(img_y + img_h, max(y, y1 + 10))

            self.crop_coords = (x1, y1, x2, y2)
        
        elif self.dragging_rect:
            # Mover el rectángulo completo
            dx = x - self.drag_start_pos[0]
            dy = y - self.drag_start_pos[1]
            
            x1, y1, x2, y2 = self.drag_start_coords
            new_x1 = x1 + dx
            new_y1 = y1 + dy
            new_x2 = x2 + dx
            new_y2 = y2 + dy
            
            # Verificar si el rectángulo sigue dentro de la imagen
            if new_x1 < img_x:
                dx = img_x - x1
            if new_y1 < img_y:
                dy = img_y - y1
            if new_x2 > img_x + img_w:
                dx = (img_x + img_w) - x2
            if new_y2 > img_y + img_h:
                dy = (img_y + img_h) - y2
            
            # Aplicar el desplazamiento
            self.crop_coords = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
        
        self.draw_crop_rectangle()

    def on_touch_up(self, touch):
        """Finaliza la operación de arrastre."""
        self.dragging_corner = None
        self.dragging_rect = False
        self.drag_start_pos = None
        self.drag_start_coords = None

    def crop_frame(self, frame):
        """Recorta un frame según las coordenadas actuales."""
        if not self.crop_coords or not self.image_widget.texture:
            return frame

        # Obtener región de visualización real de la imagen
        img_x, img_y, img_w, img_h = self.get_image_display_region()
        
        # Si las coordenadas están fuera de la región, devolver el frame completo
        if not (img_x <= self.crop_coords[0] <= img_x + img_w and 
                img_y <= self.crop_coords[1] <= img_y + img_h):
            return frame
        
        # Convertir coordenadas de pantalla a coordenadas de imagen
        x1, y1, x2, y2 = self.crop_coords
        img_height, img_width = frame.shape[:2]
        
        # Convertir las coordenadas relativas a la región de visualización a coordenadas de píxeles
        rel_x1 = (x1 - img_x) / img_w
        rel_y1 = 1.0 - (y2 - img_y) / img_h  # Invertir Y para OpenCV
        rel_x2 = (x2 - img_x) / img_w
        rel_y2 = 1.0 - (y1 - img_y) / img_h  # Invertir Y para OpenCV
        
        # Convertir a píxeles en la imagen original
        pixel_x1 = int(rel_x1 * img_width)
        pixel_y1 = int(rel_y1 * img_height)
        pixel_x2 = int(rel_x2 * img_width)
        pixel_y2 = int(rel_y2 * img_height)
        
        # Limitar a los bordes de la imagen
        pixel_x1 = max(0, min(pixel_x1, img_width - 1))
        pixel_y1 = max(0, min(pixel_y1, img_height - 1))
        pixel_x2 = max(0, min(pixel_x2, img_width - 1))
        pixel_y2 = max(0, min(pixel_y2, img_height - 1))
        
        # Asegurar que x1 < x2 e y1 < y2
        pixel_x1, pixel_x2 = min(pixel_x1, pixel_x2), max(pixel_x1, pixel_x2)
        pixel_y1, pixel_y2 = min(pixel_y1, pixel_y2), max(pixel_y1, pixel_y2)
        
        return frame[pixel_y1:pixel_y2, pixel_x1:pixel_x2]
    
    def get_image_display_region(self):
        """Calcula la región real donde se muestra la imagen en la pantalla."""
        if not self.image_widget or not self.image_widget.texture:
            return 0, 0, 0, 0
            
        # Obtener dimensiones del widget y la textura
        widget_width, widget_height = self.image_widget.size
        texture_width, texture_height = self.image_widget.texture.size
        
        # Calcular relación de aspecto
        texture_ratio = texture_width / texture_height
        widget_ratio = widget_width / widget_height
        
        # Calcular dimensiones reales de la imagen mostrada
        if texture_ratio > widget_ratio:  # Imagen más ancha que widget
            display_width = widget_width
            display_height = widget_width / texture_ratio
            x_offset = 0
            y_offset = (widget_height - display_height) / 2
        else:  # Imagen más alta que widget
            display_height = widget_height
            display_width = widget_height * texture_ratio
            y_offset = 0
            x_offset = (widget_width - display_width) / 2
            
        # Calcular la posición absoluta en la pantalla
        abs_x = self.image_widget.x + x_offset
        abs_y = self.image_widget.y + y_offset
        
        return abs_x, abs_y, display_width, display_height
    
    def initialize_crop_rectangle(self):
        """Inicializa el rectángulo de recorte para que cubra toda la imagen."""
        x, y, w, h = self.get_image_display_region()
        if w > 0 and h > 0:
            self.crop_coords = (x, y, x + w, y + h)
            self.draw_crop_rectangle()
    
    def toggle_crop_mode(self):
        """Activa/desactiva el modo de recorte"""
        self.crop_mode = not self.crop_mode
        self.crop_mode_button_color = [0, 1, 0, 1] if self.crop_mode else [0, 0.5, 1, 1]
    
        if not self.crop_mode:
            self.root.get_screen('edit').ids.video_box.canvas.after.clear()
        elif self.image_widget.texture:
            if self.crop_coords is None:
                self.initialize_crop_rectangle()
            else:
                self.draw_crop_rectangle()

    def save_crop_settings(self):
        """Guarda los ajustes de recorte, aplica recorte temporal y pasa a la siguiente pantalla."""
    
        # Crear una carpeta "temp" si no existe
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
        # Rutas de los archivos temporales
        cropped_output_path = os.path.join(temp_dir, "temp_cropped_video.mp4")
        final_output_path = os.path.join(temp_dir, "final_video.mp4")
    
        # Iniciar procesamiento del video
        self.cap = cv2.VideoCapture(self.selected_video)

        # Obtener un frame de muestra para determinar las dimensiones
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ret, sample_frame = self.cap.read()
        if not ret:
            self.show_error_dialog("No se pudo leer el video.")
            return

        # Probar el recorte para obtener dimensiones
        cropped_sample = self.crop_frame(sample_frame)
        height, width = cropped_sample.shape[:2]

        if width <= 0 or height <= 0:
            self.show_error_dialog("Área de recorte inválida. Asegúrate de que sea visible.")
            return

        # Reiniciar la posición del video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # Configurar el codec y el writer para el video cropeado
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        cropped_writer = cv2.VideoWriter(cropped_output_path, fourcc, fps, (width, height))

        # Procesar cada frame en el rango seleccionado y aplicar recorte
        current_frame = self.start_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        while current_frame <= self.end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            cropped_frame = self.crop_frame(frame)
            cropped_writer.write(cropped_frame)

            current_frame += 1

        cropped_writer.release()
        self.cap.release()

        # Aplicar cortes de inicio y fin usando cut_start_time y cut_end_time
        self.cap = cv2.VideoCapture(cropped_output_path)
        cut_writer = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))

        cut_start_frame = int(self.cut_start_time * fps)
        cut_end_frame = int(self.cut_end_time * fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cut_start_frame)

        current_frame = cut_start_frame

        while current_frame <= cut_end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            cut_writer.write(frame)
            current_frame += 1

        cut_writer.release()
        self.cap.release()

        # Guardar la ruta del video final
        self.selected_video = final_output_path

        if self.crop_mode:
            self.toggle_crop_mode()  # Desactiva el modo crop

        # Pasar a la pantalla de parámetros
        self.root.current = "parameters"

    def on_video_state_change(self, state):
        """Se llama cuando cambia el estado del video"""
        print(f"Estado del video cambiado a: {state}")
        self.is_playing = (state == 'play')

    def format_time(self, seconds):
        """Formatea segundos a MM:SS"""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def cut_video(self):
        if self.cut_start_time >= self.cut_end_time:
            self.show_error_dialog("Error: El punto de inicio debe ser menor que el punto final.")
            return
    
        # Formatear tiempos para mostrar
        start_time_formatted = self.format_time(self.cut_start_time)
        end_time_formatted = self.format_time(self.cut_end_time)
    
        print(f"Recorte desde {self.cut_start_time:.2f}s ({start_time_formatted}) hasta {self.cut_end_time:.2f}s ({end_time_formatted})")
        #Snackbar(text=f"Video recortado de {start_time_formatted} a {end_time_formatted}").open()
    
        self.root.current = "parameters"
    def generate_dynamic_boxes(self, cortes):
        # Limpia las cajas existentes si hay alguna
        dynamic_boxes = self.root.get_screen('result').ids.dynamic_boxes
        dynamic_boxes.clear_widgets()

        # Genera un MDTextField para cada valor en cortes
        for index, value in enumerate(cortes):
            text_field = MDTextField(
                text=str(value),
                hint_text=f"Valor {index + 1}",
                size_hint_y=None,
                height=dp(48),
                mode="rectangle"
            )
            dynamic_boxes.add_widget(text_field)    
    def process_video(self):
        try:
            # Crear una barra de progreso como widget personalizado
            self.progress_bar = MDProgressBar(
                value=0,
                max=100,
                type="determinate",
                size_hint_y=None,
                height=dp(48)
            )
        
            # Mostrar diálogo de progreso usando content_cls
            self.progress_dialog = MDDialog(
                title="Procesando video",
                type="custom",
                content_cls=self.progress_bar,  # Cambiado de 'content' a 'content_cls'
                auto_dismiss=False
            )
            self.progress_dialog.open()
        
        # Comenzar procesamiento en el próximo frame para permitir que se muestre el diálogo
            Clock.schedule_once(self.start_processing, 0.1)
        except Exception as e:
            error_message = f"Error al iniciar procesamiento: {str(e)}"
            self.show_error_dialog(error_message)
            print(error_message)
    
    def start_processing(self, dt):
        # Crear un hilo para ejecutar la tarea en segundo plano
        threading.Thread(target=self.process_video_task).start()

    def process_video_task(self):
        try:
            # Aquí todo tu código de procesamiento original,
            # incluyendo el manejo de progreso y actualizaciones
            params = self.root.get_screen('parameters')
        
            # Preparar parámetros
            crop_percent = params.ids.crop_percent.value
            keep_first_original = params.ids.keep_first_original.active
            keep_last_original = params.ids.keep_last_original.active
            interval = int(params.ids.interval.value)
            filtro = params.ids.filtro_button.text
            metodo_comparacion = params.ids.metodo_comparacion_button.text
            umbral_coincidencia = params.ids.umbral_coincidencia.value
            incremento_paso = int(params.ids.incremento_paso.value)
            escala_reduccion = params.ids.escala_reduccion.value
            self.metodo_fusion = params.ids.metodo_fusion_button.text
        
            # Actualizar progreso (25%)
            Clock.schedule_once(lambda dt: self.update_progress(25))
        
            # Obtener frames del video
            self.processed_frames = get_frames(
                self.selected_video, 
                crop_percent, 
                keep_first_original, 
                keep_last_original, 
                interval
            )
        
            Clock.schedule_once(lambda dt: self.update_progress(50))

            # Procesar frames
            self.cascade_frames, cascade_keypoints, kp_inverse, self.direccion = cascade_frame_search(self.processed_frames)
        
            Clock.schedule_once(lambda dt: self.update_progress(75))
        
            # Obtener valores para unión
            cortes = getValores(
                self.cascade_frames, 
                filtro, 
                metodo_comparacion, 
                umbral_coincidencia, 
                incremento_paso, 
                escala_reduccion, 
                self.direccion
            )
        
            # Generar cajas dinámicas en el hilo principal
            Clock.schedule_once(lambda dt: self.generate_dynamic_boxes(cortes))
            Clock.schedule_once(lambda dt: self.update_progress(90))
        
            # Unir frames
            imagen_fusionada = unir_cascade_frames_manteniendo_superior(
                self.cascade_frames, 
                cortes, 
                self.metodo_fusion, 
                self.direccion
            )
        
            # Guardar la imagen fusionada en un archivo temporal
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file_path = temp_file.name
                cv2.imwrite(temp_file_path, imagen_fusionada)
        
            self.processed_image_path = temp_file_path
        
            # Actualizar la UI en el hilo principal
            Clock.schedule_once(lambda dt: self.update_progress(100))
            Clock.schedule_once(lambda dt: setattr(self.root.get_screen('result').ids.result_image, "source", self.processed_image_path))
            #Clock.schedule_once(lambda dt: self.root.get_screen('result').ids.result_image.source = self.processed_image_path)
        
            # Cambiar a la pantalla de resultados
            Clock.schedule_once(lambda dt: self.progress_dialog.dismiss())
            Clock.schedule_once(lambda dt: setattr(self.root, 'current', "result"))
        
        except Exception as e:
            # Manejo de excepciones en el hilo secundario
            Clock.schedule_once(lambda dt: self.progress_dialog.dismiss())
            Clock.schedule_once(lambda dt: self.show_error_dialog(f"Error durante el procesamiento: {str(e)}"))
            import traceback
            traceback.print_exc()
    def update_progress(self, value):
    # Actualizar la barra de progreso de forma segura
        self.progress_dialog.content_cls.value = value

    def regenerate_image(self):
        # Recoge los valores editados de los MDTextField
        dynamic_boxes = self.root.get_screen('result').ids.dynamic_boxes
        nuevos_cortes = [float(box.text) for box in dynamic_boxes.children if box.text]
        nuevos_cortes.reverse()
        # Llama nuevamente a unir_cascade_frames_manteniendo_superior con los valores editados
        imagen_fusionada = unir_cascade_frames_manteniendo_superior(
            self.cascade_frames,
            nuevos_cortes,
            self.metodo_fusion,
            self.direccion
        )

        # Actualiza la imagen en la pantalla
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, imagen_fusionada)
        self.root.get_screen('result').ids.result_image.source = temp_file_path
    
    def save_result(self):
        if not self.processed_image_path:
            self.show_error_dialog("No hay imagen procesada para guardar")
            return
        
        save_path = self.settings.get("image_path", "")
        if not save_path:

            # Crear el FileChooser dentro de un Popup
            filechooser = FileChooserIconView(dirselect=True)
        
            # Crear un botón "Aceptar"
            accept_button = Button(text="Aceptar",size_hint=(0.3, 0.1))
            accept_button.bind(on_release=lambda x: self.on_save_location_selected(filechooser,popup=popup))

            # Crear un layout para el Popup
            layout = BoxLayout(orientation='vertical')
            layout.add_widget(filechooser)
            layout.add_widget(accept_button)

            # Crear el Popup
            popup = Popup(title="Selecciona una ubicación para guardar",
                      content=layout,
                      size_hint=(0.9, 0.9))
            popup.open()
        else:
            try:
                shutil.copy(self.processed_image_path, save_path)
                self.show_success_dialog("Imagen guardada exitosamente")
            except FileNotFoundError:
                self.show_error_dialog("La imagen procesada no existe o no se encontró")
            except PermissionError:
                self.show_error_dialog("No se tienen permisos suficientes para guardar la imagen")
            except Exception as e:
                self.show_error_dialog(f"Error inesperado al guardar la imagen: {str(e)}")
            toast("Imagen guardada exitosamente en: " + save_path)

    def on_save_location_selected(self, filechooser, popup):
        selection = filechooser.selection
        if selection:
            save_path = selection[0]  # Obtener la ruta seleccionada
            try:
                # Aquí debes implementar la lógica para guardar la imagen
                shutil.copy(self.processed_image_path, save_path)
                self.show_success_dialog("Imagen guardada exitosamente")
            except FileNotFoundError:
                self.show_error_dialog("La imagen procesada no existe o no se encontró")
            except PermissionError:
                self.show_error_dialog("No se tienen permisos suficientes para guardar la imagen")
            except Exception as e:
                self.show_error_dialog(f"Error inesperado al guardar la imagen: {str(e)}")
                print("Error al guardar la imagen:", e)
        popup.dismiss()
        toast("Imagen guardada exitosamente en: " + save_path)

    def show_success_dialog(self, message):
        # Implementa tu lógica para mostrar un diálogo de éxito
        pass

    def new_project(self):
        # Reiniciar todo y volver a la pantalla inicial
        self.selected_video = ""
        self.processed_image_path = ""
        self.processed_frames = []
        self.root.get_screen('video_selection').ids.selected_video_label.text = "Ninguno"
        self.root.get_screen('video_selection').ids.next_button.disabled = True
        self.root.get_screen('video_selection').ids.filechooser.selection = []
        self.root.current = "video_selection"
    
    def go_to_settings(self):
        self.root.current = "settings"
        settings_screen = self.root.get_screen("settings")
        # Sincronizar el estado del interruptor
        theme_switch = settings_screen.ids.theme_switch
        theme_switch.active = self.settings.get("theme", "Light") == "Dark"
    
    def go_back(self):
        current_screen = self.root.current
        if current_screen == "settings":
            self.root.current = "video_selection"
        elif current_screen == "edit":
            self.root.current = "video_selection"
        elif current_screen == "parameters":
            self.root.current = "edit"
        elif current_screen == "result":
            self.root.current = "parameters"
    
    def show_error_dialog(self, message):
        dialog = MDDialog(
            title="Error",
            text=message,
            buttons=[
                MDFlatButton(
                    text="ACEPTAR",
                    radius= [dp(8)],
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()
    
    def show_info_dialog(self):
        dialog = MDDialog(
            title="Información",
            text="Esta aplicación permite procesar videos para crear efectos de panorámicas temporales. Selecciona un video, configura los parámetros y disfruta de los resultados.",
            buttons=[
                MDFlatButton(
                    text="CERRAR",
                    radius= [dp(8)],
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

if __name__ == "__main__":
    VideoProcessingApp().run()
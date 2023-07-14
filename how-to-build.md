## Prerrequisitos
Si aún no lo has hecho, instala [Android Studio](https://developer.android.com/studio/index.html), siguiendo las instrucciones en el sitio web.

* Android Studio 2022.2.1 o superior.
* Un dispositivo Android o un emulador de Android con más de 4 GB de memoria.

## Compilación y ejecución con Android Studio
* Abre Android Studio y, desde la pantalla de bienvenida, selecciona "Abrir un proyecto de Android Studio existente".
* En la ventana "Abrir archivo o proyecto" que aparece, navega hasta el directorio `lite/examples/generative_ai/android` donde clonaste el repositorio de muestras de TensorFlow Lite.
* Es posible que también necesites instalar varias plataformas y herramientas según los mensajes de error.
* Cambia el nombre del modelo convertido `.tflite` a `autocomplete.tflite` y cópialo en la carpeta `app/src/main/assets/`.
* Selecciona el menú "Construir -> Compilar proyecto" para compilar la aplicación (Ctrl+F9, dependiendo de tu versión).
* Haz clic en el menú "Ejecutar -> Ejecutar 'app'" (Shift+F10, dependiendo de tu versión).

Alternativamente, también puedes utilizar el [wrapper de Gradle](https://docs.gradle.org/current/userguide/gradle_wrapper.html#gradle_wrapper) para compilarlo en la línea de comandos. Consulta la [documentación de Gradle](https://docs.gradle.org/current/userguide/command_line_interface.html) para obtener más información.

## (Opcional) Construcción del archivo .aar
De forma predeterminada, la aplicación descarga automáticamente los archivos .aar necesarios. Pero si deseas construir los tuyos propios, cambia al directorio `app/libs/build_aar/` y ejecuta `./build_aar.sh`. Este script obtendrá las operaciones necesarias de [TensorFlow Text](https://www.tensorflow.org/text) y construirá el archivo .aar para [Select TF operators](https://www.tensorflow.org/lite/guide/ops_select).

Después de la compilación, se generará un nuevo archivo `tftext_tflite_flex.aar`. Reemplaza el archivo `.aar` en la carpeta `app/libs/` y vuelve a compilar la aplicación.

Ten en cuenta que aún debes incluir el archivo .aar estándar de `tensorflow-lite` en tu archivo de Gradle.
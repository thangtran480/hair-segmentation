package com.thangtv.hairapp.model

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import androidx.annotation.RequiresApi
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.io.FileWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.logging.Logger


@Suppress("DEPRECATION")
class Segmentation (private val assetManager: AssetManager){

    companion object{
        const val SEGMENTATION_MODEL_PATH = "model_hairnet.tflite"
        const val NUMBER_THREAD = 4
        const val THRESHOLD = 0.25f
        const val NORMALIZED = true
        const val QUANTIZED = false
        const val BATCH_SIZE = 1
        const val PIXEL_SIZE = 3
        const val IMAGE_MEAN = 128
        const val IMAGE_STD = 128.0f
        const val INPUT_WIDTH = 224
        const val INPUT_HEIGHT = 224
        const val OUTPUT_WIDTH = 224
        const val OUTPUT_HEIGHT = 224
    }

    private var interpreter : Interpreter? = null
    private val log = Logger.getLogger(this::javaClass.name)


    init {
        val delegate = GpuDelegate()
        val options = Interpreter.Options().addDelegate(delegate)
        options.setNumThreads(NUMBER_THREAD)
        this.interpreter = this.loadModel()?.let { Interpreter(it, options) }
    }

    /**
     * Load model in folder assets
     */
    private fun loadModel(): MappedByteBuffer? {
        val fileDescriptor = assetManager.openFd(SEGMENTATION_MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @RequiresApi(Build.VERSION_CODES.KITKAT)
    fun run(bitmap: Bitmap){
        val tik = System.currentTimeMillis()

        val input = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, false)
        val byteBuffer = this.bitmapToByteBuffer(input)

        val output = Array(1){Array(OUTPUT_HEIGHT){Array(OUTPUT_WIDTH){FloatArray(1)} } }

        this.interpreter?.run(byteBuffer, output)

        log.warning("Run: time run: ${System.currentTimeMillis() - tik}")

        postProcessOutput(output)
    }

    @RequiresApi(Build.VERSION_CODES.KITKAT)
    private fun postProcessOutput(output: Array<Array<Array<FloatArray>>>, saveImageToTxt: Boolean = false) {

        var pixel = ""
        for( i in 0 until OUTPUT_HEIGHT){
            for (j in 0 until OUTPUT_WIDTH){
                pixel += output[0][i][j][0].toString() + " "
            }
            pixel += "\n"
        }

        if (saveImageToTxt){
            val file = File( Environment.getExternalStorageDirectory().absolutePath + "/Download", "text.txt")
            if (!file.exists()){
                file.createNewFile()
            }
            log.warning("Save File: ${file.absolutePath}")
            val writer = FileWriter(file)
            writer.append(pixel)
            writer.flush()
            writer.close()

        }
    }

    /**
     * Convert file bitmap to bytebuffer type in order to put into Tensorflow model
     */
    private fun bitmapToByteBuffer(
        bitmap: Bitmap
    ): ByteBuffer {
        val byteBuffer: ByteBuffer = if (QUANTIZED) {
            ByteBuffer.allocateDirect(BATCH_SIZE * INPUT_HEIGHT* INPUT_WIDTH* PIXEL_SIZE)
        } else {
            ByteBuffer.allocateDirect(4 * BATCH_SIZE * INPUT_HEIGHT * INPUT_WIDTH * PIXEL_SIZE)
        }

        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_HEIGHT * INPUT_WIDTH)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (pixel in 0 until INPUT_HEIGHT * INPUT_WIDTH) {
            val value = intValues[pixel]

            if (QUANTIZED) {
                byteBuffer.put((value shr 16 and 0xFF).toByte())
                byteBuffer.put((value shr 8 and 0xFF).toByte())
                byteBuffer.put((value and 0xFF).toByte())
            } else {
                if (NORMALIZED) {
                    byteBuffer.putFloat(((value shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((value shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                    byteBuffer.putFloat(((value and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                } else {
                    byteBuffer.putFloat((value shr 16 and 0xFF).toFloat())
                    byteBuffer.putFloat((value shr 8 and 0xFF).toFloat())
                    byteBuffer.putFloat((value and 0xFF).toFloat())
                }
            }
        }
        return byteBuffer
    }

}
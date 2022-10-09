/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.aohasoft.fooddetection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Typeface
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import com.aohasoft.fooddetection.R
import com.aohasoft.fooddetection.customview.OverlayView
import com.aohasoft.fooddetection.env.BorderedText
import com.aohasoft.fooddetection.env.ImageUtils
import com.aohasoft.fooddetection.env.Logger
import com.aohasoft.fooddetection.tflite.YoloV4Classifier
import com.aohasoft.fooddetection.tracking.MultiBoxTracker
import com.aohasoft.fooddetection.tflite.Classifier
import java.io.IOException
import java.io.InputStream
import java.util.LinkedList
import java.util.Locale

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity() {

    private lateinit var trackingOverlay: OverlayView
    private var sensorOrientation: Int = 0
    private lateinit var detector: Classifier
    private lateinit var rgbFrameBitmap: Bitmap
    private lateinit var croppedBitmap: Bitmap
    private lateinit var cropCopyBitmap: Bitmap
    private var computingDetection = false
    private var timestamp: Long = 0
    private lateinit var frameToCropTransform: Matrix
    private lateinit var cropToFrameTransform: Matrix
    private lateinit var tracker: MultiBoxTracker
    private lateinit var borderedText: BorderedText

    private val pricesTextView: TextView by lazy { findViewById(R.id.prices) }
    private val pricesMap: MutableMap<String, Float> = mutableMapOf()

    override fun initialize() {
        val pricesList = mutableListOf<Float>()
        val pricesInput: InputStream = assets.open(TF_OD_API_PRICE_FILE)
        pricesInput.bufferedReader().useLines { sequence ->
            pricesList.addAll(sequence.map { it.toFloat() })
        }
        pricesInput.close()

        val labelsList = mutableListOf<String>()
        val labelsInput: InputStream = assets.open(TF_OD_API_LABELS_FILE)
        labelsInput.bufferedReader().useLines { sequence ->
            labelsList.addAll(sequence)
        }
        labelsInput.close()
        pricesMap.clear()
        pricesMap.putAll(labelsList.zip(pricesList).toMap())
    }

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        val textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics)
        borderedText = BorderedText(textSizePx)
        borderedText.setTypeface(Typeface.MONOSPACE)
        tracker = MultiBoxTracker(this)
        var cropSize = TF_OD_API_INPUT_SIZE
        try {
            detector = YoloV4Classifier.create(
                    assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE)
            cropSize = TF_OD_API_INPUT_SIZE
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing classifier!")
            val toast = Toast.makeText(
                    applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT)
            toast.show()
            finish()
        }
        previewWidth = size.width
        previewHeight = size.height
        sensorOrientation = rotation - screenOrientation
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation)
        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)
        cropToFrameTransform = Matrix()
        frameToCropTransform = ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropSize, cropSize,
                sensorOrientation, MAINTAIN_ASPECT).apply {
            invert(cropToFrameTransform)
        }
        trackingOverlay = (findViewById<View>(R.id.tracking_overlay) as OverlayView).apply {
            addCallback { canvas: Canvas? -> tracker.draw(canvas) }
        }
        tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation)
    }

    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true
        LOGGER.i("Preparing image $currTimestamp for detection in bg thread.")
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight)
        readyForNextImage()
        val canvas = Canvas(croppedBitmap)
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap)
        }
        runInBackground {
            LOGGER.i("Running detection on image $currTimestamp")
            val results = detector.recognizeImage(croppedBitmap)
            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap)
            val canvas1 = Canvas(cropCopyBitmap)
            val paint = Paint()
            paint.color = Color.RED
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = 2.0f
            val mappedRecognitions: MutableList<Classifier.Recognition> = LinkedList()
            for (result in results) {
                val location = result.location
                if (location != null && result.confidence >= MINIMUM_CONFIDENCE_TF_OD_API) {
                    canvas1.drawRect(location, paint)
                    cropToFrameTransform.mapRect(location)
                    result.location = location
                    mappedRecognitions.add(result)
                }
            }
            tracker.trackResults(mappedRecognitions, currTimestamp)
            updatePrices(mappedRecognitions.map { it.title })
            trackingOverlay.postInvalidate()
            computingDetection = false
        }
    }

    override fun getLayoutId(): Int {
        return R.layout.tfe_od_camera_connection_fragment_tracking
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }

    private fun updatePrices(resultTitles: List<String>) {
        val totalPrice = pricesMap.filterKeys { it in resultTitles }.values.sum()
        val text = resultTitles.joinToString { "${it.replace("_", " ").capitalize(Locale.US)}: ${pricesMap[it]} TL " } + "\n\nTotal Price: $totalPrice TL"
        runOnUiThread {
            pricesTextView.text = text
        }
    }

    fun onButtonClick(view: View) {
        AlertDialog.Builder(this).setMessage(pricesTextView.text.toString().replace(", ", "\n")).setTitle("Total Price").create().show()
    }

    companion object {
        private val LOGGER = Logger()
        private const val TF_OD_API_INPUT_SIZE = 416
        private const val TF_OD_API_MODEL_FILE = "yolov4.tflite"
        private const val TF_OD_API_LABELS_FILE = "labels.txt"
        private const val TF_OD_API_PRICE_FILE = "prices.txt"
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private const val MAINTAIN_ASPECT = false
        private val DESIRED_PREVIEW_SIZE = Size(640, 480)
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }
}

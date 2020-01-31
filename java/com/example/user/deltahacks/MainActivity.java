package com.example.user.deltahacks;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.AsyncTask;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    //Load the tensorflow inference library
    static {
        System.loadLibrary("tensorflow_inference");
    }

    //PATH TO OUR MODEL FILE AND NAMES OF THE INPUT AND OUTPUT NODES
    private String MODEL_PATH = "file:///android_asset/neural_network2.pb";
    private String INPUT_NAME = "input_1";
    private String OUTPUT_NAME = "output_1";
    private TensorFlowInferenceInterface tf;

    //ARRAY TO HOLD THE PREDICTIONS AND FLOAT VALUES TO HOLD THE IMAGE DATA
    float[] PREDICTIONS = new float[1000];
    private float[] floatValues;
    private int[] INPUT_SIZE = {224,224,3};


    TextView resultView;

    private Button btnCapture;
    private Button btnGallery;
    private ImageView imgCapture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tf = new TensorFlowInferenceInterface(getAssets(),MODEL_PATH);

        btnGallery = (Button)findViewById(R.id.btnGallery);
        imgCapture = (ImageView) findViewById(R.id.capturedImage);
        resultView = (TextView) findViewById(R.id.textView) ;
        btnGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, 1);
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        Uri selectedImage = data.getData();

        if(requestCode == 1 && resultCode == RESULT_OK && data != null){
            try{
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                imgCapture.setImageBitmap(bitmap);
                predict(bitmap);
            }catch (Exception e){
                e.printStackTrace();
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }
        }

    }

    public Object[] argmax(float[] array){


        int best = -1;
        float best_confidence = 0.0f;

        for(int i = 0;i < array.length;i++){

            float value = array[i];

            if (value > best_confidence){

                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best,best_confidence};

    }



    public void predict(final Bitmap bitmap){


        //Runs inference in background thread
        new AsyncTask<Integer,Integer,Integer>(){

            @Override

            protected Integer doInBackground(Integer ...params){

                //Resize the image into 224 x 224
                Bitmap resized_image = ImageUtils.processBitmap(bitmap,224);

                //Normalize the pixels
                floatValues = ImageUtils.normalizeBitmap(resized_image,224,127.5f,1.0f);

                //Pass input into the tensorflow
                tf.feed(INPUT_NAME,floatValues,1,224,224,3);

                //compute predictions
                tf.run(new String[]{OUTPUT_NAME});

                //copy the output into the PREDICTIONS array
                tf.fetch(OUTPUT_NAME,PREDICTIONS);

                //Obtained highest prediction
                Object[] results = argmax(PREDICTIONS);


                int class_index = (Integer) results[0];
                float confidence = (Float) results[1];


                try{

                    final String conf = String.valueOf(confidence * 100).substring(0,5);



                    //Convert predicted class index into actual label name
                    final String label = ImageUtils.getLabel(getAssets().open("label.json"),class_index);



                    //Display result on UI
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultView.setText(label + " : " + conf + "%");
                            System.out.println("THIS IS LABEL: " + label);
                        }
                    });

                }

                catch (Exception e){
                    e.printStackTrace();
                }


                return 0;
            }



        }.execute(0);

    }


}

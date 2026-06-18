
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    session
)

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

import google.generativeai as genai

import os
import time



app = Flask(__name__)

app.secret_key = "riaz_secret_key_change_me"



UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):

    os.makedirs(UPLOAD_FOLDER)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



# Azure Credentials


AZURE_ENDPOINT = os.getenv(

    "AZURE_ENDPOINT"

)

AZURE_KEY = os.getenv(

    "AZURE_KEY"

)



client = ImageAnalysisClient(

    endpoint=AZURE_ENDPOINT,

    credential=AzureKeyCredential(

        AZURE_KEY

    )

)



# Gemini


GEMINI_API_KEY=os.getenv(

    "GEMINI_API_KEY"

)


genai.configure(

api_key=GEMINI_API_KEY

)




model = genai.GenerativeModel(

    "gemini-2.5-flash-lite"

)




def explain_image(

    caption,

    objects,

    tags

):


    prompt=f"""

You are an image assistant.

Caption:

{caption}

Objects:

{objects}

Tags:

{tags}

Explain the image in ONLY 2-3 sentences.

Maximum 50 words.

Do not use bullet points.

Write naturally.

"""


    response=model.generate_content(

        prompt

    )


    return response.text




def chat_with_image(

    question,

    caption,

    objects,

    tags

):


    prompt=f"""

You are an AI visual assistant.


Image Caption:

{caption}


Objects:

{objects}


Tags:

{tags}



User Question:

{question}



Answer naturally.

Maximum 60 words.

"""


    response=model.generate_content(

        prompt

    )


    return response.text





@app.route("/")

def home():


    return render_template(

        "index.html"

    )



@app.route(

    "/analyze",

    methods=["POST"]

)

def analyze():

    try:


        start=time.time()


        image=request.files["image"]


        filepath=os.path.join(

            app.config["UPLOAD_FOLDER"],

            image.filename

        )


        image.save(

            filepath

        )



        with open(

            filepath,

            "rb"

        ) as f:


            image_data=f.read()



        result=client.analyze(

            image_data=image_data,

            visual_features=[

                VisualFeatures.CAPTION,

                VisualFeatures.OBJECTS,

                VisualFeatures.TAGS

            ]

        )



        caption="No caption"


        objects=[]

        tags=[]



        if result.caption:


            caption=result.caption.text



        if isinstance(

            result.objects,

            dict

        ):


            for obj in result.objects["values"]:


                for tag in obj["tags"]:


                    objects.append(

                        {

                            "name":

                            tag["name"],

                            "confidence":

                            round(

                                tag["confidence"],

                                2

                            )

                        }

                    )



    
        if isinstance(

            result.tags,

            dict

        ):


            for tag in result.tags["values"]:


                tags.append(

                    {

                        "name":

                        tag["name"],

                        "confidence":

                        round(

                            tag["confidence"],

                            2

                        )

                    }

                )



        obj_names=[]


        for obj in objects:


            obj_names.append(

                obj["name"]

            )



        tag_names=[]


        for tag in tags:


            tag_names.append(

                tag["name"]

            )




        gemini_response=explain_image(

            caption,

            obj_names,

            tag_names

        )



        session["caption"]=caption

        session["objects"]=obj_names

        session["tags"]=tag_names




        processing_time=round(

            time.time()-start,

            2

        )




        return render_template(

            "index.html",

            caption=caption,

            objects=objects,

            tags=tags,

            image=image.filename,

            gemini_response=gemini_response,

            processing_time=processing_time

        )



    except Exception as e:


        return f"Error : {str(e)}"





@app.route(

    "/ask",

    methods=["POST"]

)

def ask():



    question=request.form["question"]



    caption=session.get(

        "caption",

        ""

    )



    objects=session.get(

        "objects",

        []

    )



    tags=session.get(

        "tags",

        []

    )



    answer=chat_with_image(

        question,

        caption,

        objects,

        tags

    )



    return answer





@app.route(

    "/uploads/<filename>"

)

def uploaded_file(

    filename

):



    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )




if __name__=="__main__":

    app.run(

        debug=True

    )

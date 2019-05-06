from flask import Flask, jsonify, request
import boto3  # Connecting to AWS server
import botocore  # For handling AWS error
import PyPDF2  # reading PDF files
import docx2txt  # reading word documents
import re  # for finding patterns of Phone number and email address
import os  # for removing files from local machine
import spacy  # NLP model for finding entities

# Flask App Instance
app = Flask(__name__)

# Connecting AWS Server
BUCKET_NAME = 'highporesume'  # Name of the bucket on AWS server which contains resumme
s3 = boto3.client('s3', aws_access_key_id='AKIAVAN3GMNKFS3MVASQ',
                  aws_secret_access_key='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')


# Reading resumes from local machine
def read_resume(filepath):
    """
    :param filepath: Path of the resume (make sure that it contains file extension also). Currently only PDF and DOCX file are supported
    :return: A string which contains the text from the given document. A None object is returned if file is not supported.
    """
    extension = re.search(r"\.\w+", filepath).group()
    if extension == '.pdf':
        pdfFile = open(filepath, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFile)
        resume_text = []
        for i in range(pdfReader.numPages):
            resume_text.append(pdfReader.getPage(i).extractText())
        pdfFile.close()
        return " \n ".join(resume_text)
    elif extension == '.docx':
        return docx2txt.process(filepath)
    else:
        return None


# Function to parse  new resume
def get_resume_entities(text):
    """
    :param text: Takes a string consist of text from resume.
    :return: A dictionary (object) with keys as entities labels and value as entities text as a list
    """
    results = {}
    nlp_skills = spacy.load("./resume_parser_skills")
    nlp_personal = spacy.load("./resume_parser_v1")
    # nlp = spacy.load("en")

    try:
        doc_skills = nlp_skills(text)
        doc_personal = nlp_personal(text)
    except UnicodeEncodeError:
        return {"error": "Unicode error occured."}

    phone_pattern = re.compile("\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}", re.I)
    email_pattern = re.compile("[\w\.-]+@[\w\.-]+", re.I)

    phone = phone_pattern.findall(text.replace("\n", " "))
    email = email_pattern.findall(text.replace("\n", " "))

    if phone:
        results["contact_details"] = phone
    if email:
        results["email"] = email

    for ent in doc_personal.ents:
        results[ent.label_] = []
    for ent in doc_personal.ents:
        results[ent.label_].append(ent.text)

    for ent in doc_skills.ents:
        results[ent.label_] = []
    for ent in doc_skills.ents:
        results[ent.label_].append(ent.text)

    # Removing duplicates
    for ents, values in results.items():
        results[ents] = list(set(values))

    return results


@app.route('/parse_resume', methods=['POST'])
def parse_resume():
    data = request.get_json(force=True)  # data received from post method is stored in this variable
    file_name = data['file_url'].split('/')[-1]
    file_extension = re.search(r"\.\w+", file_name).group()

    # Download the file into local machine
    try:
        s3.download_file(BUCKET_NAME, file_name, 'FILE_NAME' + file_extension)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    resume_text = read_resume('FILE_NAME' + file_extension)
    os.remove('FILE_NAME' + file_extension)

    results = get_resume_entities(resume_text)

    responses = jsonify(results=results)
    responses.status_code = 200
    return (responses)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port {}".format(port))
    app.run(debug=False, port=port, host='0.0.0.0')

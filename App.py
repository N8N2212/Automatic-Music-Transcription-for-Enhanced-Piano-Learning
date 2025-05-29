import os
from flask import Flask, request, render_template, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

from Transcription import transcribe_audio_file

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
RESULTS_FOLDER = os.path.join(os.getcwd(), "results")
ALLOWED_EXTENSIONS = {"mp3", "wav"}

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    # Render the upload form
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    # Check for file in request
    if "file" not in request.files:
        flash("No file part in the request.")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(input_path)
        
        try:
            # Create output PDF name based on the uploaded file
            base_name = os.path.splitext(filename)[0]
            output_pdf = os.path.join(RESULTS_FOLDER, f"{base_name}.pdf")
            # Call transcribe_audio_file once and capture the returned filename
            actual_pdf = transcribe_audio_file(input_path, output_pdf, show_debug_plots=False)
            print("Transcription output file:", actual_pdf)
        except Exception as e:
            flash(f"Error during transcription: {e}")
            return redirect(url_for("index"))
        
        # Redirect to the view page so the user can see the PDF inline
        return redirect(url_for("view_file", filename=os.path.basename(actual_pdf)))
    else:
        flash("File type not allowed. Only mp3 and wav files are accepted.")
        return redirect(request.url)

# Route to serve the PDF file (inline)
@app.route("/results/<filename>")
def results_file(filename):
    return send_file(os.path.join(RESULTS_FOLDER, filename), mimetype="application/pdf")

# Route to display the PDF in an embedded viewer
@app.route("/view/<filename>")
def view_file(filename):
    return render_template("view_pdf.html", filename=filename)

if __name__ == "__main__":
    # Run without debug reloading to prevent unwanted restarts during long transcription
    app.run(debug=False, use_reloader=False)

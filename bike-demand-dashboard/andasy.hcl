# andasy.hcl app configuration file generated for machine-learning on Tuesday, 31-Mar-26 11:55:26 CAT
#
# See https://github.com/quarksgroup/andasy-cli for information about how to use this file.

app_name = "machine-learning"

app {

  env = {
    PYTHON_VERSION            = "3.11"
    SQLITE_PATH               = "/data/app.db"
    UPLOAD_FOLDER             = "/data/uploads"
    SAVED_MODELS_FOLDER       = "/data/saved_models"
    EXPORTS_FOLDER            = "/data/exports"
    SEND_FILE_MAX_AGE_DEFAULT = "300"
  }

  port = 8080

  primary_region = "kgl"

  compute {
    cpu      = 1
    memory   = 2048
    cpu_kind = "shared"
  }

  process {
    name = "machine-learning"
  }

  storage {
    name        = "machine-learning-data"
    destination = "/data"
  }

}

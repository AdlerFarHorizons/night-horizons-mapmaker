# We use postgis as a base image
FROM postgis/postgis

# The base image does not include any CLI tools, so we install them
RUN apt-get update -y
RUN apt-get install postgis -y
RUN apt-get clean
RUN rm -rf /var/cache/apt/lists
# For developers
## For VS code users

This directory includes configuration files for using VS code Remote Containers + Docker.
Each subdirectory contains a Dockerfile as well as a configuration file for VS code.

For instance, to use the Python 3.9 environment, you can do

```bash
ln -s .dev/container_py39 .devcontainer
```

and open the top directory of the repository by VS code.

## Fans of other editors
You can create a container from a Dockerfile or simply read it; the dockerfile
is a recipe describing setting up your environment.

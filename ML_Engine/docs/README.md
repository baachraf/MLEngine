# ML Modules Documentation

This directory contains the source files for the ML Modules documentation, which is built using [Sphinx](https://www.sphinx-doc.org/).

## Prerequisites

- Python 3.x
- pip

## First-Time Setup

Before building the documentation for the first time, you need to install Sphinx.

```sh
pip install sphinx
```

## How to Build the Documentation

1.  Navigate to this directory (`docs`) in your terminal.

    ```sh
    cd path/to/your/project/ML_Engine/docs
    ```

2.  Run the build command.

    **On Windows:**
    ```sh
    .\make.bat html
    ```

    **On macOS/Linux:**
    ```sh
    make html
    ```

3.  The generated HTML documentation will be located in the `build/html` directory. You can view the main page by opening `build/html/index.html` in your web browser.

---

## How to Update Documentation After Adding New Code

If you add new Python modules (`.py` files) to the `ml_core` library, you should first automatically generate the `.rst` source files for them before building the HTML.

1.  Navigate to this `docs` directory.

2.  Run `sphinx-apidoc` to scan for new modules. The `-f` flag forces it to overwrite existing files.
    ```sh
    sphinx-apidoc -f -M -o source/ ../
    ```

3.  Re-build the HTML with the new content.
    ```sh
    .\make.bat html
    ```

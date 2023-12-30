# Titanic



## Getting started

this is a portoflio project made by Kamil Skowronek [fullstack python developer].
This project can be used to see the data and analisys from titanic dataset, feel free to use it!




## Git 
-   first
```
cd existing_repo
git init
git remote add origin https://gitlab.com/K.Skylark/titanic
git branch -M main
<!-- git checkout <branch-name> -->
```

-   Clone
```
git clone https://gitlab.com/K.Skylark/titanic
```

-   Push
```
git push -uf origin main
```

-   Pull
```
git pull origin main
```


## Installation
Create your enviroment via Conda or Pip

- Conda:
    ```conda create --name <env_name> --file requirements.txt```

- Pip:
    ```pip install -r requirements.txt```

## Usage
- Streamlit:
    - start server ``` streamlit run app.py ```

- Docker:
    - build Docker image ``` docker build -t titanic:latest . ```
    - run Docker image ``` docker run -p 8080:8080 titanic:latest ```



## Contact
- email: kamilskylark@gmail.com
- linkedin: https://www.linkedin.com/in/kamil-skowronek-910204185/

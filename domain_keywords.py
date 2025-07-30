# domain_keywords.py

TECH_KEYWORDS = {
    'data_science': {
        'python', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'spark', 'sql', 'machine learning', 'deep learning', 'nlp',
        'computer vision', 'statistics', 'data visualization', 'mlops',
        'data engineering', 'predictive modeling', 'statistical modeling',
        'power bi', 'tableau', 'r', 'sas', 'julia', 'matlab', 'excel',
        'apache hadoop', 'hive', 'pig', 'kafka', 'etl', 'data warehousing',
        'big data', 'nosql', 'mongodb', 'cassandra', 'redis', 'airflow',
        'dbt', 'azure data factory', 'aws glue', 'google dataflow',
        'data governance', 'data quality', 'cloud computing', 'aws', 'azure',
        'gcp', 'databricks', 'snowflake', 'fivetran', 'segment', 'stream processing',
        'batch processing', 'data lakes', 'data marts', 'c++', 'scala', 'flink',
        'data modeling', 'business intelligence', 'looker', 'quicksight', 'sagemaker',
        'datarobot', 'h2o.ai'
    },
    'ai_ml': {  # New category for specific AI/ML focus
        'neural networks', 'convolutional neural networks', 'recurrent neural networks',
        'transformers', 'gpt', 'llm', 'generative ai', 'reinforcement learning',
        'supervised learning', 'unsupervised learning', 'transfer learning',
        'feature engineering', 'model evaluation', 'hyperparameter tuning',
        'model deployment', 'explainable ai', 'mlflow', 'kubeflow',
        'scikit-learn', 'xgboost', 'lightgbm', 'keras', 'opencv', 'cuda', 'gpu',
        'natural language processing', 'image recognition', 'speech recognition',
        'time series analysis', 'anomaly detection', 'computer vision', 'mlops',
        'responsible ai', 'prompt engineering', 'hugging face', 'langchain',
        'vector databases', 'pinecone', 'chroma', 'faiss', 'whisper', 'dall-e',
        'stable diffusion', 'computer linguistics'
    },
    'java_fullstack': {
        'java', 'spring', 'spring boot', 'microservices', 'docker', 'kubernetes',
        'aws', 'rest api', 'sql', 'nosql', 'ci/cd', 'testing', 'junit', 'mockito',
        'hibernate', 'jpa', 'maven', 'gradle', 'jenkins', 'git', 'github',
        'frontend', 'backend', 'fullstack', 'html', 'css', 'javascript',
        'react', 'angular', 'vue.js', 'typescript', 'webpack', 'babel',
        'springboot', 'graphql', 'apache kafka', 'rabbitmq', 'redis', 'caching',
        'security', 'oauth2', 'jwt', 'linux', 'unix', 'bash', 'design patterns',
        'agile', 'scrum', 'j2ee', 'servlets', 'jsp', 'tomcat', 'weblogic', 'websphere',
        'spring security', 'spring cloud', 'kotlin', 'gcp', 'azure', 'jenkins', 'travis ci',
        'sonarqube', 'selenium', 'cypress', 'ansible', 'terraform', 'zookeeper', 'elk stack'
    },
    'python_fullstack': {
        'python', 'django', 'flask', 'fastapi', 'rest api', 'graphql',
        'html', 'css', 'javascript', 'react', 'angular', 'vue.js', 'typescript',
        'sql', 'nosql', 'postgresql', 'mysql', 'mongodb', 'redis', 'celery',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'git', 'github',
        'jenkins', 'ansible', 'terraform', 'pytest', 'unittest', 'pydantic', 'sqlalchemy', 'oauth2', 'jwt',
        'linux', 'nginx', 'gunicorn', 'uwsgi', 'message queues', 'rabbitmq', 'kafka',
        'web sockets', 'channels', 'microservices', 'design patterns', 'agile', 'scrum',
        'graphene', 'alembic', 'psycopg2', 'werkzeug', 'jinja2', 'bootstrap', 'tailwind css',
        'cypress', 'selenium', 'robot framework', 'prometheus', 'grafana'
    },
    'mern_stack': {
        'mongodb', 'express.js', 'react.js', 'node.js', 'javascript', 'html', 'css',
        'rest api', 'api development', 'frontend development', 'backend development',
        'fullstack development', 'redux', 'context api', 'react hooks', 'next.js',
        'gatsby', 'graphql', 'apollo client', 'apollo server', 'jwt', 'authentication',
        'authorization', 'bcrypt', 'passport.js', 'mongoose', 'sequelize', 'webpack',
        'babel', 'npm', 'yarn', 'git', 'github', 'heroku', 'netlify', 'vercel',
        'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'testing', 'jest',
        'enzyme', 'react testing library', 'web sockets', 'socket.io', 'real-time applications',
        'scss', 'less', 'tailwind css', 'bootstrap', 'material ui', 'ui/ux', 'responsive design',
        'npm scripts', 'axios', 'fetch api', 'styled components', 'redux thunk', 'redux saga'
    },
    'software_eng': {  # General software engineering, useful if specific stack is unknown
        'java', 'spring', 'microservices', 'docker', 'kubernetes',
        'aws', 'rest api', 'sql', 'nosql', 'ci/cd', 'testing', 'python', 'javascript',
        'c++', 'c#', '.net', 'devops', 'agile', 'scrum', 'git', 'cloud computing',
        'system design', 'data structures', 'algorithms', 'api design', 'backend', 'frontend',
        'fullstack', 'database management', 'linux', 'cloud native', 'cybersecurity',
        'software development life cycle', 'sdlc', 'oop', 'functional programming',
        'version control', 'code review', 'continuous integration', 'continuous delivery'
    }
}

    # Add remaining categories (ai_ml, java_fullstack, python_fullstack, mern_stack, software_eng) as per your full list...


TECH_KEYWORDS_EXPANDED = {
    'python': {'python', 'py', 'pythonic', 'pythonista'},
    'machine learning': {'ml', 'ai', 'deep learning', 'neural networks', 'machinelearning'},
    'deep learning': {'dl', 'deeplearning', 'neural networks', 'cnns', 'rnns'},
    'sql': {'sql', 'structured query language', 'mysql', 'postgresql', 'sqlite', 'ms sql', 'transact-sql'},
    'docker': {'docker', 'containers', 'containerization', 'dockerize'},
    'kubernetes': {'kubernetes', 'k8s', 'container orchestration'},
    'aws': {'aws', 'amazon web services', 's3', 'ec2', 'lambda', 'rds', 'vpc', 'iam', 'cloudwatch'},
    'azure': {'azure', 'microsoft azure', 'azure devops', 'azure functions'},
    'gcp': {'gcp', 'google cloud platform', 'google cloud'},
    'ci/cd': {'ci/cd', 'continuous integration', 'continuous delivery', 'devops pipelines'},
    'javascript': {'javascript', 'js', 'ecmascript'},
    'react': {'react', 'reactjs', 'react.js'},
    'node.js': {'node', 'nodejs', 'node.js'},
    'mongodb': {'mongodb', 'mongo', 'nosql database'},
    'git': {'git', 'github', 'gitlab', 'bitbucket', 'version control'},
    'rest api': {'rest api', 'restful api', 'api', 'application programming interface'},
    'kafka': {'kafka', 'apache kafka', 'message queue'},
    'spark': {'spark', 'apache spark'},
    'tensorflow': {'tensorflow', 'tf'},
    'pytorch': {'pytorch', 'torch'},
    'nlp': {'nlp', 'natural language processing'},
    'computer vision': {'cv', 'computervision', 'image processing'},
    'data engineering': {'de', 'dataeng', 'data pipeline', 'etl', 'elt'},
    'mlops': {'mlops', 'ml deployment', 'machine learning operations'},
    'containerization': {'docker', 'kubernetes', 'containers'},
    'agile': {'agile', 'scrum', 'kanban', 'sprint'},
    'testing': {'testing', 'qa', 'quality assurance', 'test driven development', 'tdd', 'unit testing',
                'integration testing', 'e2e testing'},
    'cloud computing': {'cloud', 'aws', 'azure', 'gcp', 'cloud infrastructure'}
}

BASE_STOP_KEYWORDS_FILTER = {
    "a", "an", "the", "and", "or", "in", "on", "at", "with", "for", "from", "to", "of",
    "experience", "skills", "ability", "responsibilities", "requirements", "knowledge",
    "work", "manage", "develop", "build", "implement", "design", "ensure", "lead",
    "collaborate", "strong", "proven", "excellent", "deep", "understanding", "good",
    "relevant", "key", "successful", "solution", "system", "tools", "data", "learning",
    "machine", "vision", "computer", "algorithm", "learn", "find out", "get", "know",
    "see", "study", "take", "teach", "watch", "acquire", "ascertain", "check", "determine",
    "discover", "explore", "feel", "get a line", "get wind", "get word", "go through", "have",
    "inquiry", "instruct", "larn", "pick up", "receive", "research", "simple machine", "train",
    "arrangement", "car", "motorcar", "organisation", "organization", "scheme", "visual modality",
    "visual sensation", "visual sense", "imagination", "imaginativeness", "encyclopaedism",
    "encyclopedism", "eruditeness", "erudition", "scholarship", "learnedness", "memorise",
    "memorize", "automotive", "job description", "etc", "e.g.", "i.e.", "ability to",
    "responsible for", "working with", "experience with", "proficiency in", "strong background",
    "familiarity with", "demonstrated ability", "manage projects", "drive innovation",
    "strategic thinking", "problem solving", "critical thinking", "communication skills",
    "team player", "leadership", "complex problems", "technical skills", "business acumen",
    "client facing", "stakeholder management"
}

KEYWORD_CATEGORIES = {
    'Programming Languages': {'python', 'java', 'javascript', 'c++', 'c#', 'r', 'scala', 'go', 'kotlin', 'php', 'ruby',
                              'swift', 'typescript'},
    'Frameworks & Libraries': {'spring', 'django', 'flask', 'react', 'angular', 'vue.js', 'node.js', 'express.js',
                               'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'hadoop', 'spark', 'pandas', 'numpy',
                               'springboot', 'hibernate', 'jpa', 'redux', 'next.js', 'fastapi', 'matplotlib',
                               'seaborn', 'bokeh'},
    'Databases': {'sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'cassandra', 'redis', 'sqlite', 'oracle',
                  'sql server', 'dynamodb', 'cosmosdb'},
    'Cloud Platforms': {'aws', 'azure', 'gcp', 'heroku', 'netlify', 'vercel', 'digitalocean', 'alibaba cloud'},
    'DevOps & Tools': {'docker', 'kubernetes', 'git', 'github', 'gitlab', 'jenkins', 'ansible', 'terraform', 'jira',
                       'confluence', 'maven', 'gradle', 'webpack', 'babel', 'npm', 'yarn', 'selenium', 'cypress',
                       'jest', 'pytest', 'splunk', 'grafana', 'prometheus', 'elk stack', 'apache kafka', 'rabbitmq',
                       'airflow', 'dbt'},
    'AI/ML Concepts': {'machine learning', 'deep learning', 'nlp', 'computer vision', 'ai', 'neural networks',
                       'reinforcement learning', 'supervised learning', 'unsupervised learning', 'mlops',
                       'generative ai', 'llm', 'transformers', 'time series', 'anomaly detection'},
    'Data & Analytics': {'data engineering', 'data science', 'big data', 'etl', 'data warehousing', 'data lakes',
                         'data visualization', 'tableau', 'power bi', 'looker', 'snowflake', 'databricks', 'fivetran',
                         'segment', 'stream processing', 'batch processing'},
    'Methodologies': {'agile', 'scrum', 'kanban', 'devops', 'sdlc', 'tdd', 'bdd', 'ci/cd'},
    'Soft Skills': {'communication', 'leadership', 'problem solving', 'teamwork', 'critical thinking',
                    'analytical skills', 'strategic thinking', 'client facing'}
}

# resume

# Gijin Kim

**Contact**: abjin69@gmail.com | 010-7548-1106 | [GitHub](https://github.com/abjin) | [Velog](https://velog.io/@abjin)

## About

I'm Gijin Kim, a developer who enjoys simplifying complex world problems and solving them with IT technology. From designing backend systems that stably handle large-scale traffic to developing RAG-based intelligent services, I have focused on practical problem-solving through various technical challenges. Recently, I've been studying natural language processing technology to solve more diverse problems.

## 🎓 Education

**Dankook University**, Yongin (Mar 2020 ~ Current)
- Major: B.S. in Mobile Systems Engineering
- GPA: 4.0 / 4.5
- Relevant Coursework: Data Structures, Database, Probability and Statistics, Linear Algebra, Computer Architecture, System Programming, Operating Systems

## 💼 Work Experience

### Nudge Healthcare (Cashwalk), Seoul (Apr 2022 – Apr 2024, 2 years 1 month)
**Backend Part Leader & Backend Developer**

Responsible for technical leadership and team management including scrum facilitation, code reviews, API design, and RDB/NoSQL optimization across projects.

#### Large-Scale Chat Data Processing System Development (Chat SAAS Backend Development)
**Key Achievements and Responsibilities:**
- Designed and developed server architecture to handle large-scale data streams for 400K daily active users (DAU)
- Developed various messaging features including read receipts, 1-on-1 chat, group chat, and channel talk (simultaneous delivery to 5M+ users)
- Implemented API design based on Node.js and Nest.JS, and real-time communication using Socket.io
- Implemented multi-tenant architecture applicable to various services to ensure system scalability
- Improved data processing performance through RDB/NoSQL (MySQL, DynamoDB, Redis) design and optimization
- As part leader, conducted scrum sessions, code reviews, and schedule/task management for the backend team

#### Global Service Backend and Data Pipeline Development (Global Pedometer-based Reward Service)
**Key Achievements and Responsibilities:**
- Designed and developed backend systems for global services targeting North America and Europe
- Built stable architecture considering network latency and data consistency in global environments
- Maintained and operated features in AWS cloud environment (EC2, RDS, S3, SQS, Lambda, etc.)
- As part leader, conducted scrum sessions, code reviews, and schedule/task management for the backend team

## 🚀 Project Experience

### AI-Powered Slack Q&A Bot Development (AI-Powered Slack RAG Bot)
**Duration**: May 2025 - Jul 2025 (Personal Project)

**Description**: Designed and developed a RAG architecture-based Slack bot to build a question-answering system specialized for internal team knowledge. Built a system that provides accurate and evidence-based answers to user questions through vector search targeting GitHub codebase and Notion documents.

**Key Features and Implementation:**
- **Question Classification and Routing**: Implemented a workflow using LangGraph to analyze user question intent and dynamically route to appropriate services for each purpose
- **Vector Embedding & Search**: Used OpenAI Embedding Model to convert GitHub and Notion text data into vectors, stored in Pinecone vector DB for similarity search
- **Automatic Embedding Updates**: Implemented cron jobs using NestJS Schedule module to automatically detect latest changes in GitHub repositories every hour and update Pinecone DB embeddings
- **Multi-Tenant Architecture**: Designed DB schema and logic supporting multi-tenancy for multiple Slack workspaces to use independent information

**Tech Stack**: NestJS, TypeScript, LangChain, LangGraph, OpenAI API, Pinecone, Prisma, MySQL, Slack API, GitHub API, Notion API

### AI-Based Clothing Recycling Platform Development (AI-based Clothing Recycling Platform Server)
**Duration**: Feb 2025 - Apr 2025 (Personal Project)

**Description**: Developed a service that automates the clothing recycling sorting process using YOLO models. Improved recycling efficiency with 15-category classification and defect detection features, and implemented fast and scalable services through TorchScript optimization and Docker/AWS-based deployment pipeline.

**Key Features and Responsibilities:**
- **AI Model Serving API Development**: Developed API endpoints using FastAPI to receive clothing image URLs and return classification and defect detection results
- **Model Optimization and Serving**: Optimized inference performance by converting PyTorch models to TorchScript and built serving pipeline for real-time image processing
- **Cloud-Based Deployment Automation**: Designed CI/CD pipeline to build Docker images, push to Docker Hub, and automatically deploy to AWS Elastic Beanstalk
- **Infrastructure Development and Operations**: Built reverse proxy configuration and stable service operation environment in local and production environments using Docker Compose and Nginx

**Tech Stack**: Python, FastAPI, PyTorch, TorchScript, OpenCV, Docker, AWS Elastic Beanstalk, Nginx, Next.js

### Gym Attendance Motivation App Development (Gym Attendance Motivation App)
**Duration**: Feb 2025 - Apr 2025 (Personal Project)

**Description**: Developed a fitness web/app with gym attendance and social features. Improved the accuracy of gym image verification through prompt engineering using OpenRouter AI and implemented a real-time ranking system using Redis.

**Key Features and Responsibilities:**
- **AI Image Verification**: Implemented prompt engineering with OpenRouter's AI model to determine whether user-uploaded images are actual gyms
- **Ranking and Level System**: Implemented real-time ranking using Redis and managed user rankings and scores

**Tech Stack**: NestJS, TypeScript, Node.js, MySQL, Prisma, Redis, AWS (S3), OpenRouter, Docker, React

### Food Image-Based Calorie Analysis AI Server Development (Calorie Analysis AI Server)
**Duration**: Nov 2024 - Jan 2025 (Personal Project)

**Description**: Developed an AI API server that analyzes calories and nutritional information from food photos. Fine-tuned YOLOv11 model to detect and classify food objects, building a system that provides accurate nutritional information to users.

**Key Features and Implementation:**
- **YOLOv11 Model Fine-tuning**: Fine-tuned model with 90 food class dataset to improve food detection accuracy to mAP 0.85
- **AI Model Serving Optimization**: Built efficient AI serving pipeline for real-time image processing by integrating fine-tuned YOLOv11 model into FastAPI server
- **Food Detection and Classification**: Accurately detected and classified various foods in images through model trained on 90 food classes
- **Cloud-Based Architecture**: Designed stable and scalable server infrastructure using AWS (EC2, S3, RDS) and automated deployment with Docker
- **API Design**: Designed RESTful API handling the process from image upload to nutritional information analysis results

**Tech Stack**: Python, FastAPI, PyTorch, YOLOv11, Docker, AWS (EC2, S3, RDS)

## 🛠️ Skills

| Category | Skills |
|----------|---------|
| **Programming** | TypeScript, Python, JavaScript, Node.js |
| **AI / ML** | LangChain, LangGraph, OpenAI API, Pinecone, Hugging Face, PyTorch, YOLO |
| **Backend & DevOps** | NestJS, FastAPI, MySQL, PostgreSQL, Redis, AWS (EC2, S3, RDS, DynamoDB), Docker, Next.js |

## 🔬 Research Interests

### 1. Retrieval-Augmented Generation (RAG) & Domain-Specific LLM
Fascinated by research that leverages external information sources to enhance LLM reliability and reduce hallucinations, based on hands-on experience building RAG-based bots. Currently exploring efficient search, indexing, and generation strategies for domain-specific data.

### 2. AI Agent
Building on LangGraph-based agent development, passionate about improving interaction quality in conversational AI and advancing autonomous LLM agents capable of handling complex user needs end-to-end.
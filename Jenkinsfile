pipeline {
    agent any

    stages {
        stage('Clone Github repo') {
            steps {
                script {
                    echo 'Cloning Github repo to jenkins...............'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Devmangukiya/Medical_RAG_Chatbot.git']])
                }
            }
        }
    }
}
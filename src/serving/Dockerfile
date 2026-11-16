FROM vllm/vllm-openai:v0.15.1

RUN pip install --no-cache-dir peft==0.18.1

COPY serve.sh /serve.sh
RUN chmod +x /serve.sh

EXPOSE 8000

ENTRYPOINT ["/serve.sh"]
CMD []

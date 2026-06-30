# Agent Flow Diagram


graph TD
    A[User Input] --> B(Assistant Node: llm_with_tools)
    B -->|Tool Called| C[Tools Node: Execute Tool]
    C -->|Tool Result| D(Assistant Node: llm_no_tools)
    D --> E[Generate Final Response]
    B -->|No Tool Called| E
    E --> F[Return to User]

